//#pragma GCC optimize("Ofast,unroll-loops")

#include <iostream>
#include <vector>
#include <map>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <queue>
#include <set>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <random>
#include <ctime>
#include <functional>
#include <bitset>
#include <cmath>
#include <functional>
#include <cctype>
#include <locale>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <list>
#include <stack>
#include <chrono>
#include <cstring>

using namespace std;

vector<vector<double>> mul_mat(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    assert(a[0].size() == b.size());
    vector<vector<double>> c(a.size(), vector<double>(b[0].size()));
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < c[0].size(); ++j) {
            for (int k = 0; k < a[0].size(); ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

vector<vector<double>> add_mat(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    vector<vector<double>> c(a.size(), vector<double>(a[0].size()));
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < c[0].size(); ++j) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

vector<vector<double>> had_mat(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    vector<vector<double>> c(a.size(), vector<double>(a[0].size()));
    for (int i = 0; i < c.size(); ++i) {
        for (int j = 0; j < c[0].size(); ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    return c;
}

vector<vector<double>> transpose(const vector<vector<double>>& a) {
    int r = a.size();
    int c = a[0].size();
    vector<vector<double>> b(c, vector<double>(r));
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            b[j][i] = a[i][j];
        }
    }
    return b;
}

void solve() {
    int n, m, k; cin >> n >> m >> k;
    vector<vector<vector<double>>> a(n), b(n);
    vector<string> op(n);
    vector<int> tnh(n);
    vector<pair<double, int>> rlu(n);
    vector<pair<int, int>> mul(n);
    vector<vector<int>> sum(n);
    vector<vector<int>> had(n);

    for (int t = 0; t < m; ++t) {
        cin >> op[t];
        int r, c; cin >> r >> c;
        a[t].assign(r, vector<double>(c));
        b[t].assign(r, vector<double>(c));
    }
    for (int t = m; t < n; ++t) {
        cin >> op[t];
        if (op[t] == "tnh") {
            cin >> tnh[t];
            tnh[t]--;
            a[t].assign(a[tnh[t]].size(), vector<double>(a[tnh[t]][0].size()));
            b[t].assign(a[tnh[t]].size(), vector<double>(a[tnh[t]][0].size()));
        } else if (op[t] == "rlu") {
            cin >> rlu[t].first >> rlu[t].second;
            rlu[t].first = 1.0 / rlu[t].first;
            rlu[t].second--;
            a[t].assign(a[rlu[t].second].size(), vector<double>(a[rlu[t].second][0].size()));
            b[t].assign(a[rlu[t].second].size(), vector<double>(a[rlu[t].second][0].size()));
        } else if (op[t] == "mul") {
            cin >> mul[t].first >> mul[t].second;
            --mul[t].first, --mul[t].second;
            a[t].assign(a[mul[t].first].size(), vector<double>(a[mul[t].second][0].size()));
            b[t].assign(a[mul[t].first].size(), vector<double>(a[mul[t].second][0].size()));
        } else if (op[t] == "sum") {
            int sz; cin >> sz;
            sum[t].resize(sz);
            for (int& w : sum[t]) {
                cin >> w;
                --w;
            }
            a[t].assign(a[sum[t][0]].size(), vector<double>(a[sum[t][0]][0].size()));
            b[t].assign(a[sum[t][0]].size(), vector<double>(a[sum[t][0]][0].size()));
        } else if (op[t] == "had") {
            int sz; cin >> sz;
            had[t].resize(sz);
            for (int& w : had[t]) {
                cin >> w;
                --w;
            }
            a[t].assign(a[had[t][0]].size(), vector<double>(a[had[t][0]][0].size()));
            b[t].assign(a[had[t][0]].size(), vector<double>(a[had[t][0]][0].size()));
        } else {
            assert(0);
        }
    }
    for (int t = 0; t < m; ++t) {
        for (int i = 0; i < a[t].size(); ++i) {
            for (int j = 0; j < a[t][i].size(); ++j) {
                cin >> a[t][i][j];
            }
        }
    }
    for (int t = m; t < n; ++t) {
        int r = a[t].size(), c = a[t][0].size();
        if (op[t] == "tnh") {
            int f = tnh[t];
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < c; ++j) {
                    a[t][i][j] = tanh(a[f][i][j]);
                }
            }
        } else if (op[t] == "rlu") {
            int f = rlu[t].second;
            double alpha = rlu[t].first;
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < c; ++j) {
                    if (a[f][i][j] > 0)
                        a[t][i][j] = a[f][i][j];
                    else
                        a[t][i][j] = a[f][i][j] * alpha;
                }
            }
        } else if (op[t] == "mul") {
            auto [f1, f2] = mul[t];
            a[t] = mul_mat(a[f1], a[f2]);
        } else if (op[t] == "sum") {
            for (int f : sum[t]) {
                a[t] = add_mat(a[t], a[f]);
            }
        } else if (op[t] == "had") {
            for (auto& xx : a[t]) for (auto& yy : xx) yy = 1;
            for (int f : had[t]) {
                a[t] = had_mat(a[t], a[f]);
            }
        }
    }
    for (int t = n - k; t < n; ++t) {
        int r = a[t].size(), c = a[t][0].size();
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                cin >> b[t][i][j];
            }
        }
    }
    for (int t = n - 1; t >= 0; --t) {
        int r = a[t].size(), c = a[t][0].size();
        if (op[t] == "tnh") {
            vector<vector<double>> grad(r, vector<double>(c));
            int f = tnh[t];
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < c; ++j) {
                    grad[i][j] = (1 - tanh(a[f][i][j]) * tanh(a[f][i][j])) * b[t][i][j];
                }
            }
            b[f] = add_mat(b[f], grad);
        } else if (op[t] == "rlu") {
            vector<vector<double>> grad(r, vector<double>(c));
            auto [alpha, f] = rlu[t];
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < c; ++j) {
                    grad[i][j] = b[t][i][j] * (a[f][i][j] >= 0 ? 1 : alpha);
                }
            }
            b[f] = add_mat(b[f], grad);
        } else if (op[t] == "mul") {
            auto [f1, f2] = mul[t];
            b[f1] = add_mat(b[f1], mul_mat(b[t], transpose(a[f2])));
            b[f2] = add_mat(b[f2], mul_mat(transpose(a[f1]), b[t]));
        } else if (op[t] == "sum") {
            for (int f : sum[t]) {
                b[f] = add_mat(b[f], b[t]);
            }
        } else if (op[t] == "had") {
            for (int e = 0; e < had[t].size(); ++e) {
                int f = had[t][e];
                vector<vector<double>> prod(a[had[t][0]].size(), vector<double>(a[had[t][0]][0].size(), 1));
                for (int z = 0; z < had[t].size(); ++z) {
                    if (z == e) continue;
                    prod = had_mat(prod, a[had[t][z]]);
                }
                b[f] = add_mat(b[f], had_mat(b[t], prod));
            }
        } else {
            assert(op[t] == "var");
        }
    }
    for (int t = n - k; t < n; ++t) {
        int r = a[t].size(), c = a[t][0].size();
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                cout << fixed << a[t][i][j] << " ";
            }
            cout << "\n";
        }
    }

    for (int t = 0; t < m; ++t) {
        int r = b[t].size(), c = b[t][0].size();
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                cout << fixed << b[t][i][j] << " ";
            }
            cout << "\n";
        }
    }
}

/*
4 2 2
var 2 3
var 4 2
mul 2 1
tnh 3
1 2
3
4
5
6
7
8
9
10
11
12
13
14
 */

int32_t main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.precision(8);

    int T = 1;// cin >> T;
    while (T--)
        solve();
}
