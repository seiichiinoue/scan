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

template<typename T>
struct multiset_comparator {
    bool operator()(const pair<size_t, T> &a, const pair<size_t, T> &b) {
        return a.second > b.second;
    }
};

void split_string_by(const string &str, char delim, vector<string> &elems) {
    elems.clear();
    string item;
    for (char ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}
void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems) {
    elems.clear();
    wstring item;
    for (wchar_t ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}
bool ends_with(const std::string& str, const std::string& suffix) {
    size_t len1 = str.size();
    size_t len2 = suffix.size();
    return len1 >= len2 && str.compare(len1 - len2, len2, suffix) == 0;
}

class SCANTrainer {
public:
    SCAN *_scan;
    Vocab *_vocab;
    vector<vector<size_t>> _dataset;
    vector<vector<size_t>> _validation_dataset;
    vector<int> _times;
    vector<int> _times_validation;
    unordered_map<size_t, int> _word_frequency;
    unordered_map<size_t, int> _word_frequency_validation;

    double** _logistic_Phi;
    double*** _logistic_Psi;
    double* _probs;

    int _burn_in_period;
    int _ignore_word_count;
    int _current_iter;
    int _sense_criteria_idx;
    int _vocab_criteria_idx;
    
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
        _ignore_word_count = IGNORE_WORD_COUNT;
        _current_iter = 0;
        _sense_criteria_idx = 0;
        _vocab_criteria_idx = 0;
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
        _probs = new double[_scan->_n_k];

        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Phi[t] = new double[_scan->_n_k];
            _scan->logisitc_transformation(t, _logistic_Phi[t]);
        }
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Psi[t] = new double*[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_Psi[t][k] = new double[_scan->_vocab_size];
                _scan->logisitc_transformation(t, k, _logistic_Psi[t][k]);
            }
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            _probs[k] = 0.0;
        }
        
        _sense_criteria_idx = _scan->_n_k-1;
        _vocab_criteria_idx = vocab_size-1;
        if (_word_frequency[_vocab_criteria_idx] < _ignore_word_count) {
            for (int v=vocab_size-1; v>=0; --v) {
                if (_word_frequency[v] >= _ignore_word_count) {
                    _vocab_criteria_idx = v;
                    break;
                }
            }
        }
        for (int t=0; t<_scan->_n_t; ++t) {
            _scan->_Phi[t][_sense_criteria_idx] = 0.0;
            for (int k=0; k<_scan->_n_k; ++k) {
                _scan->_Psi[t][k][_vocab_criteria_idx] = 0.0;
            }
        }
    }
    void load_documents(string filepath) {
        wifstream ifs(filepath.c_str());
        assert(ifs.fail() == false);
        wstring sentence;
        while (getline(ifs, sentence) && !ifs.eof()) {
            int doc_id = _dataset.size();
            _dataset.push_back(vector<size_t>());
            vector<wstring> words;
            split_word_by(sentence, L' ', words);
            add_document(words, doc_id);
        }
    }
    void add_document(vector<wstring> &words, int doc_id) {
        if (words.size() == 0) return;
        vector<size_t> &doc = _dataset[doc_id];
        for (auto word : words) {
            if (word.size() == 0) continue;
            size_t word_id = _vocab->add_string(word);
            doc.push_back(word_id);
            _word_frequency[word_id] += 1;
        }
    }
    void load_time_labels(string filepath) {
        wifstream ifs(filepath.c_str());
        assert(ifs.fail() == false);
        wstring time;
        while (getline(ifs, time) && !ifs.eof()) {
            _times.push_back(stoi(time));
        }
    }
    void set_num_sense(int n_k) {
        _scan->_n_k = n_k;
    }
    void set_num_time(int n_t) {
        _scan->_n_t = n_t;
    }
    void set_kappa_phi(double kappa_phi) {
        _scan->_kappa_phi = kappa_phi;
    }
    void set_kappa_psi(double kappa_psi) {
        _scan->_kappa_psi = kappa_psi;
    }
    void set_gamma_a(double gamma_a) {
        _scan->_gamma_a = gamma_a;
    }
    void set_gamma_b(double gamma_b) {
        _scan->_gamma_b = gamma_b;
    }
    void set_context_window_width(int context_window_width) {
        _scan->_context_window_width = context_window_width;
    }
    void set_burn_in_period(int burn_in_period) {
        _burn_in_period = burn_in_period;
    }
    void set_ignore_word_count(int ignore_word_count) {
        _ignore_word_count = ignore_word_count;
    }
    int get_sum_word_frequency() {
        int sum = 0;
        for (int v=0; v<_word_frequency.size(); ++v) {
            if (_word_frequency[v] >= _ignore_word_count) {
                sum += _word_frequency[v];
            }
        }
        return sum;
    }
    int get_ignore_word_count() {
        int cnt = 0;
        for (int v=0; v<_word_frequency.size(); ++v) {
            if (_word_frequency[v] < _ignore_word_count) {
                cnt++;
            }
        }
        return cnt;
    }
    void sample_z(int t) {
        _update_logistic_Phi(true);
        _update_logistic_Psi(true);
        double** logistic_psi_t = _logistic_Psi[t];
        double* logistic_phi_t = _logistic_Phi[t];
        for (int n=0; n<_scan->_num_docs; ++n) {
            if (_times[n] != t) continue;
            // calculation for $\phi^t_k$
            double* probs_n = _probs;
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] = log(logistic_phi_t[k]);
            }
            // calculation for $\prod \psi^{t, k}_wi$
            for (int k=0; k<_scan->_n_k; ++k) {
                for (int i=0; i<_scan->_context_window_width; ++i) {
                    size_t word_id = _dataset[n][i];
                    if (_word_frequency[word_id] < _ignore_word_count) {
                        continue;
                    }
                    probs_n[k] += log(logistic_psi_t[k][word_id]);
                }
            }
            // calculation of constants for softmax transformation
            double constants = 0.0;
            for (int k=0; k<_scan->_n_k; ++k) {
                constants = _logsumexp(constants, probs_n[k], (bool)(k==0));
            }
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] -= constants;
            }
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] = exp(probs_n[k]);
            }
            // random sampling from multinomial distribution and assign new sense
            int sense = sampler::multinomial((size_t)_scan->_n_k, probs_n);
            _scan->_Z[n] = sense;
        }
    }
    double _logsumexp(double x, double y, bool flg) {
        if (flg) return y; // init mode
        // if (x == y) return x + 0.69314718055; // log(2)
        if (x == y) return x + std::log(2);
        double vmin = std::min(x, y);
        double vmax = std::max(x, y);
        if (vmax > vmin + 50) {
            return vmax;
        } else {
            return vmax + std::log(std::exp(vmin - vmax) + 1.0);
        }
    }
    void sample_phi(int t) {
        // sample phi under each time $t$
        _update_logistic_Phi();
        double* mean = new double[_scan->_n_k];
        double* lmean = new double[_scan->_n_k];
        if (t == 0) {
            mean = _scan->_Phi[t+1];
            lmean = _logistic_Phi[t+1];
        } else if (t+1 == _scan->_n_t) {
            mean = _scan->_Phi[t-1];
            lmean = _logistic_Phi[t-1];
        } else {
            for (int k=0; k<_scan->_n_k; ++k) {
                mean[k] = _scan->_Phi[t-1][k] + _scan->_Phi[t+1][k];
                mean[k] *= 0.5;
                lmean[k] = _logistic_Phi[t-1][k] + _logistic_Phi[t+1][k];
                lmean[k] *= 0.5;
            }
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            if (k == _sense_criteria_idx) {
                continue;
            }
            double constants = 0;
            for (int i=0; i<_scan->_n_k; ++i) {
                constants += exp(mean[i]) * (double)(i != k);
            }
            int cnt = 0, cnt_else = 0;
            for (int n=0; n<_scan->_num_docs; ++n) {
                if (_times[n] != t) continue;
                if (_scan->_Z[n] == k) cnt++;
                else cnt_else++;
            }
            double lu, ru;
            if (cnt == 0) {
                lu = -5.0;
            } else if (cnt_else == 0) {
                ru = 5.0;
            } else {
                // random sampling of maximum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(0, lmean[k])$ 
                lu = std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt) * lmean[k];
                lu = log(constants) + log(lu) - log(1 - lu);
                // random sampling of minimum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(lmean[k], 1)$
                ru = (1.0 - lmean[k]) * (1.0 - std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt_else)) + lmean[k];
                ru = log(constants) + log(ru) - log(1 - ru);
                // scaling probabilistic variable following logistic distribution to standard normal
                lu = (lu - mean[k]) / (PI * LVAR);
                ru = (ru - mean[k]) / (PI * LVAR);
            }
            assert(lu < ru);
            double noise = sampler::truncated_normal(lu, ru);
            double sampled = mean[k] + noise * sqrt(1.0 / _scan->_kappa_phi);
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
        // sample phi under each {time $t$, sense $k$}
        _update_logistic_Psi();
        for (int k=0; k<_scan->_n_k; ++k) {
            double* mean = new double[_scan->_vocab_size];
            double* lmean = new double[_scan->_vocab_size];
            if (t == 0) {
                mean = _scan->_Psi[t+1][k];
                lmean = _logistic_Psi[t+1][k];
            } else if (t+1 == _scan->_n_t) {
                mean = _scan->_Psi[t-1][k];
                lmean = _logistic_Psi[t-1][k];
            } else {
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    mean[v] = _scan->_Psi[t-1][k][v] + _scan->_Psi[t+1][k][v];
                    mean[v] *= 0.5;
                    lmean[v] = _logistic_Psi[t-1][k][v] + _logistic_Psi[t+1][k][v];
                    lmean[v] *= 0.5;
                }
            }
            for (int v=0; v<_scan->_vocab_size; ++v) {
                if (v == _vocab_criteria_idx) {
                    continue;
                }
                if (_word_frequency[v] < _ignore_word_count) {
                    continue;
                }
                double constants = 0;
                for (int i=0; i<_scan->_vocab_size; ++i) {
                    if (_word_frequency[i] < _ignore_word_count) {
                        continue;
                    }
                    constants += exp(mean[i]) * (double)(i != v);
                }
                int cnt = 0, cnt_else = 0;
                for (int n=0; n<_scan->_num_docs; ++n) {
                    if (_times[n] != t || _scan->_Z[n] != k) continue;
                    for (int i=0; i<_scan->_context_window_width; ++i) {
                        size_t word_id = _dataset[n][i];
                        if (_word_frequency[word_id] < _ignore_word_count) {
                            continue;
                        }
                        if (word_id == v) cnt++;
                        else cnt_else++;
                    }
                }
                double lu, ru;
                if (cnt == 0) {
                    lu = -5.0;
                } else if (cnt_else == 0) {
                    ru = 5.0;
                } else {
                    // random sampling of maximum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(0, lmean[v])$ 
                    lu = std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt) * lmean[v];
                    lu = log(constants) + log(lu) - log(1 - lu);
                    // random sampling of minimum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(lmean[v], 1)$
                    ru = (1.0 - lmean[v]) * (1.0 - std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt_else)) + lmean[v];
                    ru = log(constants) + log(ru) - log(1 - ru);
                    // scaling probabilistic variable following logistic distribution to standard normal
                    lu = (lu - mean[v]) / (PI * LVAR);
                    ru = (ru - mean[v]) / (PI * LVAR);
                }
                assert(lu < ru);
                double noise = sampler::truncated_normal(lu, ru);
                double sampled = mean[v] + noise * sqrt(1.0 / _scan->_kappa_psi);
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
            _scan->logisitc_transformation(t, _logistic_Phi[t], evalue);
        }
    }
    void _update_logistic_Psi(bool evalue=false) {
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Psi[t] = new double*[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_Psi[t][k] = new double[_scan->_vocab_size];
                _scan->logisitc_transformation(t, k, _logistic_Psi[t][k], evalue);
            }
        }
    }
    double compute_log_likelihood() {
        _update_logistic_Psi(true);
        double log_pw = 0;
        for (int t=0; t<_scan->_n_t; ++t) {
            for (int n=0; n<_scan->_num_docs; ++n) {
                if (_times[n] != t) continue;
                int assigned_sense = _scan->_Z[n];
                for (int i=0; i<_scan->_context_window_width; ++i) {
                    size_t word_id = _dataset[n][i];
                    if (_word_frequency[word_id] < _ignore_word_count) {
                        continue;
                    }
                    log_pw += log(_logistic_Psi[t][assigned_sense][word_id]);
                }
            }
        }
        return log_pw;
    }
    void train(int iter=1000, string save_path ="./bin/scan.bin") {
        for (int i=0; i<iter; ++i) {
            ++_current_iter;
            for (int t=0; t<_scan->_n_t; ++t) {
                sample_z(t);
                sample_phi(t);
                sample_psi(t);
                if (iter % 50 == 0) {
                    sample_kappa();
                }
            }
            double log_pw = compute_log_likelihood();
            cout << "iter: " << _current_iter << "\tlog_likelihood: " << log_pw << endl;
            save(save_path);
        }
    }
    void save(string filename) {
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oarchive(ofs);
        oarchive << *_vocab;
        oarchive << *_scan;
        oarchive << _word_frequency;
        oarchive << _dataset;
        oarchive << _times;
        oarchive << _burn_in_period;
        oarchive << _ignore_word_count;
        oarchive << _current_iter;
        oarchive << _sense_criteria_idx;
        oarchive << _vocab_criteria_idx;
    }
    bool load(string filename) {
        std::ifstream ifs(filename);
        if (ifs.good()) {
            _vocab = new Vocab();
            _scan = new SCAN();
            boost::archive::binary_iarchive iarchive(ifs);
            iarchive >> *_vocab;
            iarchive >> *_scan;
            iarchive >> _word_frequency;
            iarchive >> _dataset;
            iarchive >> _times;
            iarchive >> _burn_in_period;
            iarchive >> _ignore_word_count;
            iarchive >> _current_iter;
            iarchive >> _sense_criteria_idx;
            iarchive >> _vocab_criteria_idx;
            return true;
        }
        return false;
    }
};

void read_data(string data_path, SCANTrainer &trainer) {
    string documents_path = data_path+"documents.txt";
    string time_labels_path = data_path+"time_labels.txt";
    trainer.load_documents(documents_path);
    trainer.load_time_labels(time_labels_path);
}

// hyper parameters flags
DEFINE_int32(num_sense, 8, "number of sense");
DEFINE_int32(num_time, 10, "number of time interval");
DEFINE_double(kappa_phi, 4.0, "initial value of kappa_phi");
DEFINE_double(kappa_psi, 10.0, "initial value of kappa_psi (fixed)");
DEFINE_double(gamma_a, 7.0, "hyperparameter of gamma prior");
DEFINE_double(gamma_b, 3.0, "hyperparameter of gamma prior");
DEFINE_int32(context_window_width, 10, "context window width");
DEFINE_int32(num_iteration, 1000, "number of iteration");
DEFINE_int32(burn_in_period, 150, "burn in period");
DEFINE_int32(ignore_word_count, 3, "threshold of low-frequency words");
DEFINE_string(data_path, "./data/transport/", "path to dataset for training");
DEFINE_string(validation_data_path, "./data/transport/", "path to dataset for validation");
DEFINE_string(save_path, "./bin/scan.model", "path to saving model");
DEFINE_bool(from_archive, false, "load archive or not");

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    SCANTrainer trainer;
    // set hyper parameter
    trainer.set_num_sense(FLAGS_num_sense);
    trainer.set_num_time(FLAGS_num_time);
    trainer.set_kappa_phi(FLAGS_kappa_phi);
    trainer.set_kappa_psi(FLAGS_kappa_psi);
    trainer.set_gamma_a(FLAGS_gamma_a);
    trainer.set_gamma_b(FLAGS_gamma_b);
    trainer.set_context_window_width(FLAGS_context_window_width);
    trainer.set_burn_in_period(FLAGS_burn_in_period);
    trainer.set_ignore_word_count(FLAGS_ignore_word_count);
    // read dataset
    read_data(FLAGS_data_path, trainer);
    // prepare model
    trainer.prepare();
    // load archive
    if (FLAGS_from_archive) {
        trainer.load(FLAGS_save_path);
    }
    // logging summary
    cout << "num of docs: " << trainer._scan->_num_docs << endl;
    cout << "sum of word freq: " << trainer.get_sum_word_frequency() << endl;
    cout << "vocab size: " << trainer._vocab->num_words() - trainer.get_ignore_word_count() << endl;
    // tarining
    trainer.train(FLAGS_num_iteration, FLAGS_save_path);
    return 0;
}