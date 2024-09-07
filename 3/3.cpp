#include <mpi.h>
#include <bits/stdc++.h>
#define ll long long int 
#define double long double

using namespace std; 

int32_t main(int argc, char **argv) {
    // Tell MPI to start
    MPI_Init(&argc, &argv); 

    //Get the rank of process and number of processes
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n, la_size ; 
    vector<double> a; 

    if(rank == 0){
        // read the input
        freopen(argv[1],"r",stdin); 
        cin>>n; a.resize(n); 
        for(int i=0; i<n; i++) cin>>a[i] ;
    }

    double start_time = MPI_Wtime();

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int elements_per_process = n/p ;
    int extra = n%p; 

    vector<int> send_counts(p), displs(p); 
    int sum = 0; 
    for(int i=0; i<p; i++){
        int sri = elements_per_process*i + min(i, extra); 
        int eri = sri + elements_per_process + (i < extra? 0: -1);
        send_counts[i] = (eri-sri+1); 
        displs[i] = sum; 
        sum += send_counts[i]; 
    }

    vector<double> la(send_counts[rank]); // local array 
    MPI_Scatterv(a.data(), send_counts.data(), displs.data(), MPI_LONG_DOUBLE, la.data(), send_counts[rank], MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    for(int i=1; i<send_counts[rank]; i++){
        la[i] += la[i-1]; 
    }
    
    MPI_Gatherv(la.data(), send_counts[rank], MPI_LONG_DOUBLE, a.data(), send_counts.data(), displs.data(), MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        for(int i=1; i<p; i++){
            for(int j=displs[i]; j<displs[i]+send_counts[i]; j++){
                a[j] += a[displs[i]-1];
            }
        }
        for(int i=0; i<n; i++){
            cout << fixed << setprecision(2) << a[i] << " ";
        }
        cout<<endl;
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // cout << fixed <<setprecision(5)<<"Time taken (excluding input/output): " << elapsed_time << " seconds." << endl;
    }
    
    // Tell MPI to end
    MPI_Finalize(); 
}
