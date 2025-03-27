import sys
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from flask import Flask, request, jsonify
from contextlib import contextmanager

eps = 0.01

cuda_code = """
__global__ void simpson_kernel(double a, int m, double h, double *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m)
    {
        double x_2i = a + 2*idx * h;
        double x_2i_1 = a + (2*idx + 1) * h;
        double x_2i_2 = a + (2*idx + 2) * h;

        double y_2i = function(x_2i);
        double y_2i_1 = function(x_2i_1);
        double y_2i_2 = function(x_2i_2);

        results[idx] = (h / 3) * (y_2i + 4*y_2i_1 + y_2i_2);
    }
}
"""

cuda.init()
device = cuda.Device(0)
app = Flask(__name__)

def add_function_to_cuda_code(text_function, cuda_code):
    result = '\n'.join([
        "__host__ __device__ double function(double x)",
        "{",
        f"    return {text_function};",
        "}\n"
    ])
    
    return result + cuda_code

@contextmanager
def gpu_ctx():
    ctx = device.make_context()
    try:
        yield ctx
    finally:
        ctx.pop()

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        problem_data = request.json
        text_function = problem_data['function']
        a = problem_data['a']
        b = problem_data['b']
        blocksize = problem_data['blocksize']

        n_min = int((b-a) / eps)
        n = (n_min // (blocksize*2)) * (blocksize*2) + (n_min % (blocksize*2)) * (blocksize*2);
        m = n // 2
        h = (b-a) / n

        block = (blocksize, 1, 1)
        grid = (m // block[0], 1, 1)

        results = np.zeros(shape=m, dtype=float)

        with gpu_ctx():
            new_cuda_code = add_function_to_cuda_code(text_function, cuda_code)

            start_event = cuda.Event()
            end_event = cuda.Event()

            start_event.record()

            results_gpu = cuda.mem_alloc(sys.getsizeof(np.float64) * m)

            mod = SourceModule(new_cuda_code)
            kernel = mod.get_function("simpson_kernel")
            kernel(np.float64(a), np.int32(m), np.float64(h), results_gpu, block=block, grid=grid)

            cuda.memcpy_dtoh(results, results_gpu)

            integral = np.sum(results)

            results_gpu.free()

            end_event.record()
            end_event.synchronize()

            total_gpu_time_ms = start_event.time_till(end_event)

        return jsonify({
            "result": integral,
            "total_gpu_time_ms": total_gpu_time_ms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)