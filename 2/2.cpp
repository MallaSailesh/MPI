#include <mpi.h>
#include <bits/stdc++.h>
#define ll long long int
#define double long double
#define THRESHOLD 2

using namespace std;

pair<double, double> square_complex(double rl, double im){
    double nr = rl*rl - im*im; // a^2 - b^2 + 2abi
    double ni = 2*rl*im;
    return {nr, ni};
}

bool MoreThanThreshold(double rl, double im, ll row, ll col){
    double val = rl*rl + im*im;
    val = sqrt(val);
    double threshold = THRESHOLD;
    return val > threshold;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    ll n, m, k;
    double cr, ci;
    vector<int> c_ans;

    if (rank == 0) {
        freopen(argv[1],"r",stdin);
        cin >> n >> m >> k;
        cin >> cr >> ci;
    }

    double start_time = MPI_Wtime();

    MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cr, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ci, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    ll size_per_process = (n * m) / p;
    ll extra = (n * m) % p;

    ll si = size_per_process * rank + min(1ll * rank, extra);
    ll ei = si + size_per_process + (rank < extra ? 1 : 0) - 1;

    // Now, calculate the initial value i.e., z0 values from si to ei
    for (ll idx = si; idx <= ei; idx++) {
        ll row = idx / m, col = idx % m;
        double z0_rl = 3; z0_rl /= (m - 1);
        z0_rl *= col; z0_rl -= 1.5;
        double z0_im = -3; z0_im /= (n - 1);
        z0_im *= row; z0_im += 1.5;

        int in_set = 1;
        for (int _ = 0; _ <= k; _++) {
            if (MoreThanThreshold(z0_rl, z0_im, row, col)) {
                in_set = 0;
                break;
            }
            pair<double, double> new_z = square_complex(z0_rl, z0_im);
            z0_rl = new_z.first + cr;
            z0_im = new_z.second + ci;
        }
        c_ans.push_back(in_set);
    }

    // Send results from all processes to rank 0
    if (rank == 0) {
        vector<int> final_ans(n * m);
        vector<int> recv_counts(p), displs(p);

        for (int i = 0; i < p; i++) {
            recv_counts[i] = size_per_process + (i < extra ? 1 : 0);
            displs[i] = (size_per_process * i) + min(i, static_cast<int>(extra));
        }

        MPI_Gatherv(c_ans.data(), c_ans.size(), MPI_INT, final_ans.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < final_ans.size(); i++) {
            if (i % m == 0 && i != 0) cout << endl;
            cout << final_ans[i] << " ";
        }
        cout << endl;
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // cout << "Time taken (excluding input/output): " << elapsed_time << " seconds." << endl;
    } else {
        MPI_Gatherv(c_ans.data(), c_ans.size(), MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }



    MPI_Finalize();
    return 0;
}