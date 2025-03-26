import aiohttp
import asyncio
import json

async def send_to_server(session, server_url, problem):
    async with session.post(server_url + "/calculate", json=problem) as response:
        if response.status != 200:
            error = await response.text()
            print(f"Error from server {server_url}: {error}")
            return None
        return await response.json()

async def process_integral_function(session, server_url, problem):
    result = await send_to_server(session, server_url, problem)
    if result is not None:
        integral = result['result']
        print(f"Function: {problem['function']}, a = {problem['a']}, b = {problem['b']}")
        print(f"I = {integral:.5f}\n")
        return result.get("total_gpu_time_ms", 0)
    return 0

async def main():
    with open("config.json", "r") as file:
        config = json.load(file)

    if len(config['servers']) == 0:
        print('The servers must be specified in config.json.')
        return 0
    
    start_time = asyncio.get_event_loop().time()
    total_gpu_time_ms = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for problem in config['problems']:
            server_url = config['servers'][len(tasks) % len(config['servers'])]
            task = asyncio.create_task(process_integral_function(session, server_url, problem))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_gpu_time_ms = sum(results)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    print(f"All integrals calculated. Total time: {total_time:.6f} seconds")
    print(f"Total GPU time: {total_gpu_time_ms:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())