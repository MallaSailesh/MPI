    #include <mpi.h>
    #include <bits/stdc++.h>
    #define ll long long int
    #define double long double

    using namespace std;

    int main(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);

        int rank, p;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        int n; 
        vector<double> matrix, matrixI;

        if(rank == 0){
            freopen(argv[1],"r",stdin); 
            cin>>n; 
            for(int i=0; i<n*n; i++){
                int row = i/n; 
                double x; cin>>x; 
                matrix.push_back(x);
                if(i == row*n + row) matrixI.push_back(1); 
                else matrixI.push_back(0); 
            }
        }   
        double start_time = MPI_Wtime();
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int rows_per_process = n/p; 
        int extra = n%p; 
        int sr = rows_per_process*rank + min(rank, extra); 
        int er = sr + rows_per_process + (rank < extra? 0: -1); 
        int crows = (er-sr+1); 
        // cout<<"Process "<<rank<<"->"<<extra<<" "<<rows_per_process<<endl; 

        vector<int> send_counts(p), displs(p); 
        int sum = 0; 
        for(int i=0; i<p; i++){
            int sri = rows_per_process*i + min(i, extra); 
            int eri = sri + rows_per_process + (i < extra? 0: -1);
            send_counts[i] = (eri-sri+1)*n; 
            displs[i] = sum; 
            sum += send_counts[i]; 
        }

        vector<double> local_A(send_counts[rank]), local_AI(send_counts[rank]); 
        MPI_Scatterv(matrix.data(), send_counts.data(), displs.data(), MPI_LONG_DOUBLE, local_A.data(), send_counts[rank], MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(matrixI.data(), send_counts.data(), displs.data(), MPI_LONG_DOUBLE, local_AI.data(), send_counts[rank], MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < n; i++) {
            int owner = (i < extra*(rows_per_process+1)) ? i/(rows_per_process+1): ((i-extra*(rows_per_process+1))/(rows_per_process))+extra;  
            vector<double> pivot_row_A(n), pivot_row_AI(n);
            // if(rank == 0) cout<<i<<" "<<owner<<endl;
            if (rank == owner) {
                int local_row = i - sr;
                double factor = local_A[local_row * n + i];

                for (int j = 0; j < n; j++) {
                    local_A[local_row * n + j] /= factor;
                    local_AI[local_row * n + j] /= factor;
                }

                pivot_row_A = vector<double>(local_A.begin() + local_row * n, local_A.begin() + (local_row + 1) * n);
                pivot_row_AI = vector<double>(local_AI.begin() + local_row * n, local_AI.begin() + (local_row + 1) * n);
            }

            MPI_Barrier(MPI_COMM_WORLD); // blocks all MPI Process until they call this routine 

            MPI_Bcast(pivot_row_A.data(), n, MPI_LONG_DOUBLE, owner, MPI_COMM_WORLD);
            MPI_Bcast(pivot_row_AI.data(), n, MPI_LONG_DOUBLE, owner, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD); 

            for (int j = 0; j < crows; j++) {
                if (sr + j == i) continue;
                
                double factor = local_A[j * n + i];
                for (int k = 0; k < n; k++) {
                    local_A[j * n + k] -= factor * pivot_row_A[k];
                    local_AI[j * n + k] -= factor * pivot_row_AI[k];
                }
            }
        }

        MPI_Gatherv(local_AI.data(), send_counts[rank], MPI_LONG_DOUBLE, matrixI.data(), send_counts.data(), displs.data(), MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

        if(rank == 0) {
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    cout << fixed << setprecision(2) << matrixI[i * n + j] << " ";
                }
                cout << endl;
            }
            double end_time = MPI_Wtime();
            double elapsed_time = end_time - start_time;

            // cout << fixed <<setprecision(5)<<"Time taken (excluding input/output): " << elapsed_time << " seconds." << endl;
        }

        MPI_Finalize();
        return 0;
    }