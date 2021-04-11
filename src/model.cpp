#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <thread>
#include <dirent.h>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map> 
#include "scan.hpp"
#include "vocab.hpp"
using namespace boost;
using namespace scan;

class SCANTrainer {
public:
    SCAN *_scan;
    Vocab *_vocab;
    vector<vector<size_t>> _dataset;
    vector<vector<size_t>> _validation_dataset;
    vector<int> _years;
    vector<int> _years_validation;
    double** _logistic_Phi;
    double*** _logistic_Psi;
    double* _probs;

    int _burn_in_period;
    int _current_iter;
    
    SCANTrainer() {
        setlocale(LC_CTYPE, "ja_JP.UTF-8");
        ios_base::sync_with_stdio(false);
        locale default_loc("ja_JP.UTF-8");
        locale::global(default_loc);
        locale ctype_default(locale::classic(), default_loc, locale::ctype);
        wcout.imbue(ctype_default);
        wcin.imbue(ctype_default);
        _scan = new SCAN();
        _vocab = new Vocab();
        _logistic_Phi = NULL;
        _logistic_Psi = NULL;
        _probs = NULL;

        _burn_in_period = BURN_IN_PERIOD;
        _current_iter = 0;
    }
    ~SCANTrainer() {
        delete _scan;
        delete _vocab;
    }
    void prepare() {
        int vocab_size = _vocab->num_words();
        int num_docs = _dataset.size();
        _scan->initialize(vocab_size, num_docs);

        _logistic_Phi = new double*[_scan->_n_t];
        _logistic_Psi = new double**[_scan->_n_t];

        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Phi[t] = new double[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _scan->logisitc_transformation(t, _logistic_Phi[t]);
            }
        }
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Psi[t] = new double*[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_Psi[t][k] = new double[_scan->_vocab_size];
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _scan->logisitc_transformation(t, k, _logistic_Psi[t][k]);
                }
            }
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            _probs[k] = 0;
        }
    }
    void sample_z(int t) {
        _update_logistic_Phi(true);
        _update_logistic_Psi(true);
        double** logistic_psi_t = _logistic_Psi[t];
        double* logistic_phi_t = _logistic_Phi[t];
        for (int n=0; n<_scan->_num_docs; ++n) {
            if (_years[n] != t) continue;
            // calculation for $\phi^t_k$
            double* probs_n = _probs;
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] = logistic_phi_t[k];
            }
            // calculation for $\prod \psi^{t, k}_wi$
            for (int k=0; k<_scan->_n_k; ++k) {
                for (int i=0; i<_scan->_context_window_width; ++i) {
                    size_t word_id = _dataset[n][i];
                    probs_n[k] *= logistic_psi_t[k][word_id];
                }
            }
            // calculation of constants
            double constants = accumulate(probs_n, probs_n+_scan->_n_k, 0);
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] /= constants;
            }
            // random sampling from multinomial distribution and assign new sense
            int sense = sampler::multinomial((size_t)_scan->_n_k, probs_n);
            _scan->_Z[n] = sense;
        }
    }
    void sample_phi(int t) {
        _update_logistic_Phi();
        double* mean = new double[_scan->_n_k];
        for (int k=0; k<_scan->_n_k; ++k) {
            mean[k] = 0.0;
        }
        if (t == 0) {
            mean = _logistic_Phi[t+1];
        } else if (t+1 == _scan->_n_t) {
            mean = _logistic_Phi[t-1];
        } else {
            for (int k=0; k<_scan->_n_k; ++k) {
                mean[k] += _logistic_Phi[t-1][k];
                mean[k] += _logistic_Phi[t+1][k];
                mean[k] *= 0.5;
            }
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            double lu = -1e15, ru = 1e15;
            double constants = 0;
            for (int i=0; i<_scan->_n_k; ++i) {
                constants += exp(_scan->_Phi[t][i]) * (double)(i != k);
            }
            for (int n=0; n<_scan->_num_docs; ++n) {
                if (_years[n] != t) continue;
                double u_n, bound;
                if (_scan->_Z[n] == k) {
                    u_n = sampler::uniform(0, mean[k]);
                    bound = log(constants) + log(u_n) - log(1 - u_n);
                    lu = max(lu, bound);
                } else {
                    u_n = sampler::uniform(mean[k], 1);
                    bound = log(constants) + log(u_n) - log(1 - u_n);
                    ru = min(ru, bound);
                }
            }
            lu -= mean[k];
            ru -= mean[k];
            double sampled = mean[k] + _scan->generate_noise_for_phi_from_truncated_normal_distribution(lu, ru) * sqrt(1.0 / _scan->_kappa_phi);
            _scan->_Phi[t][k] = sampled;
            if (_current_iter > _burn_in_period) {
                _scan->_EPhi[t][k] *= (_current_iter - _burn_in_period - 1);
                _scan->_EPhi[t][k] += _scan->_Phi[t][k];
                _scan->_EPhi[t][k] /= (_current_iter - _burn_in_period);
            }
        }
        return;
    }
    void sample_psi(int t) {
        _update_logistic_Psi();
        for (int k=0; k<_scan->_n_k; ++k) {
            double* mean = new double[_scan->_vocab_size];
            for (int v=0; v<_scan->_vocab_size; ++v) {
                mean[v] = 0.0;
            }
            if (t == 0) {
                mean = _logistic_Psi[t+1][k];
            } else if (t+1 == _scan->_n_t) {
                mean = _logistic_Psi[t-1][k];
            } else {
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    mean[v] += _logistic_Psi[t-1][k][v];
                    mean[v] += _logistic_Psi[t+1][k][v];
                    mean[v] *= 0.5;
                }
            }
            for (int v=0; v<_scan->_vocab_size; ++v) {
                double lu = -1e15, ru = 1e15;
                double constants = 0;
                for (int i=0; i<_scan->_vocab_size; ++i) {
                    constants += exp(_scan->_Psi[t][k][i]) * (double)(i != v);
                }
                for (int n=0; n<_scan->_num_docs; ++n) {
                    if (_scan->_Z[n] != k || _years[n] != t) continue;
                    double u_n, bound;
                    if (_word_in_document(v, n)) {
                        u_n = sampler::uniform(0, mean[v]);
                        bound = log(constants) + log(u_n) - log(1 - u_n);
                        lu = max(lu, bound);
                    } else {
                        u_n = sampler::uniform(mean[v], 1);
                        bound = log(constants) + log(u_n) - log(1 - u_n);
                        ru = min(ru, bound);
                    }
                }
                lu -= mean[k];
                ru -= mean[k];
                double sampled = mean[v] + _scan->generate_noise_for_psi_from_truncated_normal_distribution(lu, ru) * sqrt(1.0 / _scan->_kappa_psi);
                _scan->_Psi[t][k][v] = sampled;
                if (_current_iter > _burn_in_period) {
                    _scan->_EPsi[t][k][v] *= (_current_iter - _burn_in_period - 1);
                    _scan->_EPsi[t][k][v] += _scan->_Psi[t][k][v];
                    _scan->_EPsi[t][k][v] /= (_current_iter - _burn_in_period);
                }
            }
        }
        return;
    }
    void sample_kappa() {
        _scan->_kappa_phi = sampler::gamma(_scan->_gamma_a, _scan->_gamma_b);
        if (_current_iter > _burn_in_period) {
            _scan->_Ekappa_phi *= (_current_iter - _burn_in_period - 1);
            _scan->_Ekappa_phi += _scan->_Ekappa_phi;
            _scan->_Ekappa_phi /= (_current_iter - _burn_in_period);
        }
        return;
    }
    bool _word_in_document(size_t word_id, int doc_id) {
        vector<size_t>& tar_doc = _dataset[doc_id];
        for (int i=0; i<tar_doc.size(); ++i) {
            if (tar_doc[i] == word_id) return true;
        }
        return false;
    }
    void _update_logistic_Phi(bool evalue=false) {
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Phi[t] = new double[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _scan->logisitc_transformation(t, _logistic_Phi[t]);
            }
        }
    }
    void _update_logistic_Psi(bool evalue=false) {
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Psi[t] = new double*[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_Psi[t][k] = new double[_scan->_vocab_size];
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _scan->logisitc_transformation(t, k, _logistic_Psi[t][k]);
                }
            }
        }
    }
};

int main() {
    return 0;
}