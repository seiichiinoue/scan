#include "src/model.hpp"

bool compare_by_b(std::pair<wstring, double> a, std::pair<wstring, double> b) {
    if (a.second != b.second) {
        return a.second > b.second;
    } else {
        return a.first < b.first;
    }
}
std::vector<std::pair<wstring, double>> word_ranking(SCANTrainer &trainer, int t, int k, int top_n=10, bool prob=false) {
    std::vector<std::pair<wstring, double>> pw;
    for (int v=0; v<trainer._scan->_vocab_size; ++v) {
        if (trainer._word_frequency[v] < trainer._ignore_word_count) {
            continue;
        }
        wstring word = trainer._vocab->word_id_to_string(v);
        pw.push_back(make_pair(word, trainer._logistic_Psi[t][k][v]));
    }
    sort(pw.begin(), pw.end(), compare_by_b);
    return pw;
}
std::vector<double> sense_probability(SCANTrainer &trainer, int t) {
    std::vector<double> ps;
    for (int k=0; k<trainer._scan->_n_k; ++k) {
        ps.push_back(trainer._logistic_Phi[t][k]);
    }
    return ps;
}

int main() {
    SCANTrainer trainer;
    // load archive
    string path = "./bin/transport.model";
    bool ret = trainer.load(path);
    // prepare model
    trainer.prepare();
    trainer._update_logistic_Phi();
    trainer._update_logistic_Psi();
    // for each t, k, visualize word and sense distribution
    for (int t=0; t<trainer._scan->_n_t; ++t) {
        wcout << "time:" << t << endl;
        std::vector<double> ps = sense_probability(trainer, t);
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            wcout << "sense:" << k << "(" << ps[k] << ") ";
            std::vector<std::pair<wstring, double>> pw = word_ranking(trainer, t, k);
            for (int i=0; i<10; ++i) {
                wcout << pw[i].first << " ";
            }
            wcout << endl;
        }
    }
    return 0;
}