import os

for file in os.listdir("input"):
	os.system(f"python visualize_result.py -i input/{file} -o output/{file}")