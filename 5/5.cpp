#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long long n;
    vector<long long> arr;

    if(rank == 0)
    {
        if(argc < 2)
        {
            cerr << "Usage: " << argv[0] << " <input_file>" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ifstream infile(argv[1]);
        if (!infile.is_open())
        {
            cerr << "Error: Could not open file " << argv[1] << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile >> n;
        arr.resize(n + 1);
        for(long long i = 0; i < n + 1; i++)
        {
            infile >> arr[i];
        }
        infile.close();
    }

    MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        arr.resize(n + 1);
    }
    MPI_Bcast(arr.data(), n + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));
    vector<long long> diag(n, 0);

    for(long long len = 2; len < n + 1; len++)
    {
        if(len > 2)
        {
            diag.resize(n + 2 - len);
            for(long long i = 0; i < n - len + 2; i++)
            {
                long long j = i + len - 1;
                dp[i][j] = diag[i];
                diag[i] = 0;
            }
        }

        long long queries_per_process = (n + 1 - len) / static_cast<long long>(size);
        long long remainder = (n + 1 - len) % static_cast<long long>(size);
        long long start_index = static_cast<long long>(rank) * queries_per_process + min(static_cast<long long>(rank), remainder);
        long long end_index = start_index + queries_per_process + (static_cast<long long>(rank) < remainder ? 1 : 0);

        vector<long long> local_arr(end_index - start_index);
        for(long long i = start_index; i < end_index; i++)
        {
            long long j = i + len;
            local_arr[i - start_index] = LLONG_MAX;
            for(long long k = i + 1; k < j; k++)
            {
                long long cost = dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j];
                local_arr[i - start_index] = min(local_arr[i - start_index], cost);
            }
        }

        vector<int> recv_counts(size);
        vector<int> displacements(size);
        for (int i = 0; i < size; ++i) {
            recv_counts[i] = static_cast<int>((n + 1 - len) / size + (i < remainder ? 1 : 0));
            displacements[i] = static_cast<int>(static_cast<long long>(i) * ((n + 1 - len) / size) + min(static_cast<long long>(i), remainder));
        }

        MPI_Gatherv(local_arr.data(), static_cast<int>(local_arr.size()), MPI_LONG_LONG,
                    diag.data(), recv_counts.data(), displacements.data(),
                    MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        MPI_Bcast(diag.data(), static_cast<int>(n + 2 - len), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
        cout << diag[0] << endl;
    }

    MPI_Finalize();
    return 0;
}