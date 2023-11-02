#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <mpi.h>

// please do permute operation before using concat.
// To transpose a tensor in PyTorch from shape [A, B, C, D] to [D, A, B, C], you can use the permute method. 
// # Assuming original_tensor has shape [A, B, C, D]
// original_tensor = torch.randn(A, B, C, D)
// # Transpose it to shape [D, A, B, C]
// transposed_tensor = original_tensor.permute(3, 0, 1, 2)
// use tensor_concat along the first dim.  ----> [D_1 + D_2+  ...,  A, B, C] (new_D)
// Change it back to shape [A, B, C, D]
// original_tensor = transposed_tensor.permute(1, 2, 3, 0)

torch::Tensor tensor_concat(torch::Tensor part_tensor) {
    // Get the current MPI rank and the total number of ranks
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // Check tensor contiguity and datatype
    assert(part_tensor.is_contiguous());
    // assert(part_tensor.is_contiguous() && part_tensor.dtype() == torch::kFloat32);
    bool on_gpu = part_tensor.device().is_cuda();
    if (on_gpu) {
        part_tensor = part_tensor.cpu();
    }
    // Get the original tensor shape
    std::vector<int64_t> original_shape(part_tensor.sizes().vec());

    // Get the local size along the concat dimension
    int64_t local_dim_size = original_shape[0];

    // Calculate the total size along the concat dimension
    int64_t concat_dim_size;
    MPI_Allreduce(&local_dim_size, &concat_dim_size, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);


    // Serialize the tensor to a buffer
    std::vector<float> send_buffer(part_tensor.numel());
    std::memcpy(send_buffer.data(), part_tensor.data_ptr(), sizeof(float) * part_tensor.numel());


    // Gather the sizes of the tensors on all processes
    int send_count = send_buffer.size();
    std::vector<int> recv_counts(num_ranks);
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate the displacements for MPI_Allgatherv and the total size of the received buffer
    std::vector<int> displs(num_ranks);
    int total_size = recv_counts[0];
    displs[0] = 0;
    for (int i = 1; i < num_ranks; i++) {
        total_size += recv_counts[i];
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }

    // Gather the tensors on all processes
    std::vector<float> recv_buffer(total_size);
    MPI_Allgatherv(send_buffer.data(), send_count, MPI_FLOAT, recv_buffer.data(), recv_counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    //torch::Tensor complete_tensor = torch::from_blob(recv_buffer.data(), {total_size}).clone();

    // Deserialize the received buffer to a tensor on all processes
    original_shape[0] = concat_dim_size;
    torch::Tensor complete_tensor = torch::from_blob(recv_buffer.data(), original_shape).clone();
    if (on_gpu) {
        complete_tensor = complete_tensor.to(at::kCUDA);
    }
    return complete_tensor;
}


// torch::Tensor tensor_split_last_dim(torch::Tensor input_tensor) {
//     // Check contiguity and datatype
//     // Check tensor contiguity and datatype
//     assert(input_tensor.is_contiguous() && input_tensor.dtype() == torch::kFloat32);
 
//     bool on_gpu = input_tensor.device().is_cuda();
//     if (on_gpu) {
//         input_tensor = input_tensor.cpu();
//     }

//     // Get the size of the last dimension
//     int64_t last_dim_size = input_tensor.size(-1);

//     // Compute the size of the two halves
//     int64_t first_half_size = last_dim_size / 2;
//     int64_t second_half_size = last_dim_size - first_half_size;

//     // Split the tensor along the last dimension
//     auto result_tensors = input_tensor.split_with_sizes({first_half_size, second_half_size}, -1);

//     // For the purpose of this function, let's return the first half.
//     // In a real application, you might want to return both halves,
//     // or do something else with them.
//     if (on_gpu) {
//         result_tensors[0] = result_tensors[0].to(at::kCUDA);
//     }
//     return result_tensors[0];
// }

// Function to perform MPI_Bcast
torch::Tensor tensor_broadcast(torch::Tensor input_tensor) {
    // Check tensor contiguity and datatype
    assert(input_tensor.is_contiguous() && input_tensor.dtype() == torch::kFloat32);
    
    bool on_gpu = input_tensor.device().is_cuda();
    if (on_gpu) {
        input_tensor = input_tensor.cpu();
    }

    // The number of elements in the tensor
    int64_t num_elements = input_tensor.numel();

    // Perform MPI_Bcast
    MPI_Bcast(
        /*buffer=*/input_tensor.data_ptr<float>(), 
        /*count=*/num_elements, 
        /*datatype=*/MPI_FLOAT, 
        /*root=*/0, 
        // /*root=*/root_process, 
        /*comm=*/ MPI_COMM_WORLD
    );

    if (on_gpu) {
        input_tensor = input_tensor.to(at::kCUDA);
    }

    return input_tensor;
}

torch::Tensor tensor_allreduce(torch::Tensor input_tensor) {
    // Check tensor contiguity and datatype
    assert(input_tensor.is_contiguous() && input_tensor.dtype() == torch::kFloat32);
 
    bool on_gpu = input_tensor.device().is_cuda();
    if (on_gpu) {
        input_tensor = input_tensor.cpu();
    }

    // The number of elements in the tensor
    int64_t num_elements = input_tensor.numel();

    // Allocate buffer for the result
    std::vector<float> result_buffer(num_elements);

    // Perform MPI_Allreduce
    MPI_Allreduce(
        /*sendbuf=*/input_tensor.data_ptr<float>(), 
        /*recvbuf=*/result_buffer.data(), 
        /*count=*/num_elements, 
        /*datatype=*/MPI_FLOAT, 
        /*op=*/MPI_SUM, 
        /*comm=*/ MPI_COMM_WORLD
    );

    // Convert the result buffer back into a tensor with the same shape as the input
    torch::Tensor result_tensor = torch::from_blob(result_buffer.data(), input_tensor.sizes()).clone();
    
    if (on_gpu) {
        result_tensor = result_tensor.to(at::kCUDA);
    }
    return result_tensor;
}

void tensor_send(torch::Tensor input_tensor, int64_t dst) {
    // Check tensor contiguity and datatype
    assert(input_tensor.is_contiguous() && input_tensor.dtype() == torch::kFloat32);
 // Or use tensor = tensor.contiguous().cpu();
    bool on_gpu = input_tensor.device().is_cuda();
    if (on_gpu) {
        input_tensor = input_tensor.cpu();
    }

    // The number of elements in the tensor
    int64_t num_elements = input_tensor.numel();
 
    // Perform MPI_Send
    MPI_Send(
        /*buf=*/input_tensor.data_ptr<float>(), 
        /*count=*/num_elements, 
        /*datatype=*/MPI_FLOAT, 
        /*dest=*/dst,
        /*tag=*/0,
        /*comm=*/ MPI_COMM_WORLD
    );

}

torch::Tensor tensor_recv(int64_t src, std::vector<int64_t> shape) {
    // Number of elements in tensor
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());

    // Allocate buffer for the result
    std::vector<float> result_buffer(num_elements);

    // Perform MPI_Recv
    MPI_Status status;
    MPI_Recv(
        /*buf=*/result_buffer.data(), 
        /*count=*/num_elements, 
        /*datatype=*/MPI_FLOAT, 
        /*source=*/src,
        /*tag=*/0,
        /*comm=*/ MPI_COMM_WORLD,
        /*status=*/ &status
    );

    // Convert the result buffer back into a tensor with the same shape as the input
    torch::Tensor result_tensor = torch::from_blob(result_buffer.data(), shape).clone();

    return result_tensor;
}


TORCH_LIBRARY(mpi_extensions, m) {
  m.def("tensor_concat", tensor_concat);
  m.def("tensor_allreduce", tensor_allreduce);
  m.def("tensor_broadcast", &tensor_broadcast);
  m.def("tensor_send", &tensor_send);
  m.def("tensor_recv", &tensor_recv);
}

// torch::Tensor tensor_isend(torch::Tensor input_tensor, int64_t src, int64_t dst) {
//     // Check tensor contiguity and datatype
//     assert(input_tensor.is_contiguous() && input_tensor.dtype() == torch::kFloat32);
//  // Or use tensor = tensor.contiguous().cpu();
//     bool on_gpu = input_tensor.device().is_cuda();
//     if (on_gpu) {
//         input_tensor = input_tensor.cpu();
//     }

//     // The number of elements in the tensor
//     int64_t num_elements = input_tensor.numel();
 
// MPI_Request request;
// MPI_Status status;
// int flag = 0;

// // Perform MPI_Isend
// MPI_Isend(
//     /*buf=*/input_tensor.data_ptr<float>(), 
//     /*count=*/num_elements, 
//     /*datatype=*/MPI_FLOAT, 
//     /*dest=*/dst,
//     /*tag=*/0, 
//     /*comm=*/MPI_COMM_WORLD, 
//     /*request=*/&request
// );

// // Wait until the send operation completes
// MPI_Test(&request, &flag, &status);
// while(flag == 0){
//     MPI_Test(&request, &flag, &status);
// }

//     // Convert the result buffer back into a tensor with the same shape as the input
//     torch::Tensor result_tensor = torch::from_blob(result_buffer.data(), input_tensor.sizes()).clone();
    
//     if (on_gpu) {
//         result_tensor = result_tensor.to(at::kCUDA);
//     }
//     return result_tensor;
// }
