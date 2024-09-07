#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Point {
    long double x, y;
};

struct DistanceIndex {
    long double distance;
    int index;
};

long double distance(const Point& p1, const Point& p2) {
    return sqrtl(powl(p1.x - p2.x, 2) + powl(p1.y - p2.y, 2));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, M, K;
    vector<Point> P, Q;

    if (rank == 0) {
        if (argc < 2) {
            cerr << "Usage: " << argv[0] << " <input_file>" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        ifstream input_file(argv[1]);
        if (!input_file) {
            cerr << "Error opening file: " << argv[1] << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        input_file >> N >> M >> K;
        P.resize(N);
        Q.resize(M);

        for (int i = 0; i < N; i++) {
            input_file >> P[i].x >> P[i].y;
        }

        for (int i = 0; i < M; i++) {
            input_file >> Q[i].x >> Q[i].y;
        }

        input_file.close();
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        P.resize(N);
        Q.resize(M);
    }

    MPI_Bcast(P.data(), N * 2, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Q.data(), M * 2, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    int queries_per_process = M / size;
    int remainder = M % size;
    int start_index = rank * queries_per_process + min(rank, remainder);
    int end_index = start_index + queries_per_process + (rank < remainder ? 1 : 0);

    vector<vector<DistanceIndex>> local_k_nearest(end_index - start_index);

    for (int i = start_index; i < end_index; i++) {
        vector<DistanceIndex> all_distances(N);
        for (int j = 0; j < N; j++) {
            long double dist = distance(Q[i], P[j]);
            all_distances[j] = {dist, j};
        }

        partial_sort(all_distances.begin(), all_distances.begin() + K, all_distances.end(),
                     [](const DistanceIndex& a, const DistanceIndex& b) {
                         return a.distance < b.distance;
                     });

        local_k_nearest[i - start_index] = vector<DistanceIndex>(all_distances.begin(), all_distances.begin() + K);
    }

    if (rank == 0) {
        vector<vector<DistanceIndex>> k_nearest(M);

        for (int p = 1; p < size; p++) {
            int p_start_index = p * queries_per_process + min(p, remainder);
            int p_end_index = p_start_index + queries_per_process + (p < remainder ? 1 : 0);

            for (int i = p_start_index; i < p_end_index; i++) {
                k_nearest[i].resize(K);
                MPI_Recv(k_nearest[i].data(), K * sizeof(DistanceIndex), MPI_BYTE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int i = 0; i < end_index - start_index; i++) {
            k_nearest[i] = local_k_nearest[i];
        }

        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                int index = k_nearest[i][k].index;
                cout << P[index].x << " " << P[index].y << endl;
            }
        }


    } else {
        for (int i = 0; i < end_index - start_index; i++) {
            MPI_Send(local_k_nearest[i].data(), K * sizeof(DistanceIndex), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
