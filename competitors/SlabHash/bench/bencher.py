import subprocess
import datetime
import os
import json 
import sys 
import getopt
import csv

RESULTS_DIR = "../build/bench_result/"

def analyze_singleton_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		for trial in trials:
			data_q0 = (trial["load_factor"], trial["build_rate_mps"], trial["search_rate_mps"], trial["search_rate_bulk_mps"])

		print("===============================================================================================")
		print("Singleton experiment:")
		print("\tNumber of elements to be inserted: %d" % (trials[0]['num_keys']))
		print("\tNumber of buckets: %d" % (trials[0]['num_buckets']))
		print("\tExpected chain length: %.2f" % (trials[0]['exp_chain_length']))
		print("===============================================================================================")
		print("load factor\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		print("%.2f\t\t%.3f\t\t%.3f\t\t%.3f" % (data_q0[0], data_q0[1], data_q0[2], data_q0[3]))

def analyze_load_factor_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["load_factor"], 
				trial["build_rate_mps"], 
				trial["search_rate_mps"], 
				trial["search_rate_bulk_mps"], 
				trial['num_buckets']))

		tabular_data.sort()
		print("===============================================================================================")
		print("Load factor experiment:")
		print("\tTotal number of elements is fixed, load factor (number of buckets) is a variable")
		print("\tNumber of elements to be inserted: %d" % (trials[0]['num_keys']))
		print("\t %.2f of %d queries exist in the data structure" % (trials[0]['query_ratio'], trials[0]['num_queries']))
		print("===============================================================================================")
		print("load factor\tnum buckets\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		for pair in tabular_data:
			print("%.2f\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[4], pair[1], pair[2], pair[3]))		

def analyze_table_size_experiment(input_file, csv_out=None):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["num_keys"], 
				trial['num_buckets'],
				trial['load_factor'], 
				trial["build_rate_mps"], 
				trial["search_rate_mps"], 
				trial["search_rate_bulk_mps"]))

		tabular_data.sort()
		if csv_out is not None:
			with open(csv_out, mode='w', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				csv_writer.writerow(['num_keys', 'num_buckets', 'load_factor', 'build_rate_mps', 'search_rate_mps', 'search_rate_bulk_mps'])
				for pair in tabular_data:
					pair = [pair[0], pair[1], f"{pair[2]:.2f}", f"{pair[3]:.2f}", f"{pair[4]:.2f}", f"{pair[5]:.2f}"]
					csv_writer.writerow(pair)
		print("===============================================================================================")
		print("Table size experiment:")
		print("\tTable's expected chain length is fixed, and total number of elements is variable")
		print("\tExpected chain length = %.2f\n" % trials[0]['exp_chain_length'])
		print("\t%.2f of %d queries exist in the data structure" % (trials[0]['query_ratio'], trials[0]['num_queries']))
		print("===============================================================================================")
		print("(num keys, num buckets, load factor)\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		for pair in tabular_data:
			print("(%d, %d, %.2f)\t\t\t%10.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[1], pair[2], pair[3], pair[4], pair[5]))

def analyze_concurrent_experiment(input_file, csv_out=None):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["init_load_factor"], 
				trial['final_load_factor'], 
				trial['num_buckets'], 
				trial["initial_rate_mps"], 
				trial["concurrent_rate_mps"]))

		tabular_data.sort()
		last_row = tabular_data[-1]
		init_lf, final_lf, num_buckets, init_rate, conc_rate = last_row
		
		print("===============================================================================================")
		print("Concurrent experiment:")
		print("\tvariable load factor, fixed number of elements")
		print("\tOperation ratio: (insert, delete, search) = (%.2f, %.2f, [%.2f, %.2f])" % (trials[0]['insert_ratio'], trials[0]['delete_ratio'], trials[0]['search_exist_ratio'], trials[0]['search_non_exist_ratio']))
		print("===============================================================================================")
		print("batch_size = %d, init num batches = %d, final num batches = %d" % (trials[0]['batch_size'], trials[0]['num_init_batches'], trials[0]['num_batches']))
		print("===============================================================================================")
		print("init lf\t\tfinal lf\tnum buckets\tinit build rate(M/s)\tconcurrent rate(Mop/s)")
		print("===============================================================================================")
		print("%.2f\t\t%.2f\t\t%d\t\t%.3f\t\t%.3f" % (init_lf, final_lf, num_buckets, init_rate, conc_rate))
		
		table_size = trials[0]['num_keys']
		if csv_out is not None:
			write_header = not os.path.exists(csv_out)
			with open(csv_out, mode='a', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				if write_header:
					csv_writer.writerow(['table_size', 'final_load_factor', 'concurrent_rate_mps'])
				
				csv_writer.writerow([table_size, f"{final_lf:.2f}", f"{conc_rate:.2f}"])

def analyze_query_experiment(input_file, csv_out=None):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["init_load_factor"], 
				trial['final_load_factor'], 
				trial['num_buckets'], 
				trial["initial_rate_mps"],
				trial["search_exist_ratio"],
				trial["search_non_exist_ratio"],
				trial["concurrent_rate_mps"]))

		tabular_data.sort()
		last_row = tabular_data[-1]
		init_lf, final_lf, num_buckets, init_rate, query_exist, query_non_exist, conc_rate = last_row

		print("===============================================================================================")
		print("Query experiment:")
		print("\tvariable query exist ratio, fixed number of elements")
		print("\tOperation ratio: (insert, delete, search) = (%.2f, %.2f, [%.2f, %.2f])" % (trials[0]['insert_ratio'], trials[0]['delete_ratio'], trials[0]['search_exist_ratio'], trials[0]['search_non_exist_ratio']))
		print("===============================================================================================")
		print("batch_size = %d, init num batches = %d, final num batches = %d" % (trials[0]['batch_size'], trials[0]['num_init_batches'], trials[0]['num_batches']))
		print("===============================================================================================")
		print("init lf\t\tfinal lf\tnum buckets\tinit build rate(M/s)\tQuery Exist Ratio\tQuery rate(Mop/s)")
		print("===============================================================================================")
		print("%.2f\t\t%.2f\t\t%d\t\t%.3f\t\t\t%.2f\t\t%.3f" % (init_lf, final_lf, num_buckets, init_rate, query_exist, conc_rate))

		table_size = trials[0]['num_keys']
		if csv_out is not None:
			write_header = not os.path.exists(csv_out)
			with open(csv_out, mode='a', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				if write_header:
					csv_writer.writerow(['table_size', 'final_load_factor', 'query_exist_ratio', 'query_non_exist_ratio' , 'query_rate_mps'])
				csv_writer.writerow([table_size, f"{final_lf:.2f}", f"{query_exist:.2f}", f"{query_non_exist:.2f}", f"{conc_rate:.2f}"])

def analyze_rehash_experiment(input_file, csv_out=None):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]
		trial = trials[0]

		num_keys = trial["num_keys"]
		added_buckets = trial["added_buckets"]
		rehash_time_ms = trial["rehash_time_ms"]
		
		# Calculate throughput (Million operations per second)
		rehash_throughput = 0.0
		if rehash_time_ms > 0:
			rehash_throughput = num_keys / (rehash_time_ms * 1000.0)

		print("===============================================================================================")
		print("Rehash Experiment Results:")
		print("===============================================================================================")
		print("Number of Keys:       %d" % num_keys)
		print("Old Bucket Count:     %d" % trial["num_buckets_old"])
		print("New Bucket Count:     %d" % trial["num_buckets_new"])
		print("Added Buckets:        %d" % added_buckets)
		print("-----------------------------------------------------------------------------------------------")
		print("Old Build Time:       %.2f ms" % trial["old_build_time_ms"])
		print("Rehash Time:          %.2f ms" % rehash_time_ms)
		print("Rehash Throughput:    %.2f Mops" % rehash_throughput)
		print("===============================================================================================")

		if csv_out is not None:
			write_header = not os.path.exists(csv_out)
			with open(csv_out, mode='a', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				if write_header:
					csv_writer.writerow(['num_keys', 'old_buckets', 'added_buckets', 'new_buckets', 'old_build_time_ms', 'rehash_time_ms', 'rehash_throughput_Mops'])
				
				csv_writer.writerow([
					num_keys, 
					trial["num_buckets_old"], 
					added_buckets, 
					trial["num_buckets_new"], 
					f"{trial['old_build_time_ms']:.2f}", 
					f"{rehash_time_ms:.2f}",
					f"{rehash_throughput:.2f}"
				])

def analyze_merge_experiment(input_file, csv_out=None):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]
		trial = trials[0]

		num_keys = trial["num_keys"]
		removed_buckets = trial["removed_buckets"]
		merge_time_ms = trial["merge_time_ms"]
		
		# Calculate throughput
		merge_throughput = 0.0
		if merge_time_ms > 0:
			merge_throughput = num_keys / (merge_time_ms * 1000.0)

		print("===============================================================================================")
		print("Merge Experiment Results:")
		print("===============================================================================================")
		print("Number of Keys:       %d" % num_keys)
		print("Old Bucket Count:     %d" % trial["num_buckets_old"])
		print("New Bucket Count:     %d" % trial["num_buckets_new"])
		print("Removed Buckets:      %d" % removed_buckets)
		print("-----------------------------------------------------------------------------------------------")
		print("Old Build Time:       %.2f ms" % trial["old_build_time_ms"])
		print("Merge Time:           %.2f ms" % merge_time_ms)
		print("Merge Throughput:     %.2f Mops" % merge_throughput)
		print("===============================================================================================")

		if csv_out is not None:
			write_header = not os.path.exists(csv_out)
			with open(csv_out, mode='a', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				if write_header:
					csv_writer.writerow(['num_keys', 'old_buckets', 'removed_buckets', 'new_buckets', 'old_build_time_ms', 'merge_time_ms', 'merge_throughput_Mops'])
				
				csv_writer.writerow([
					num_keys, 
					trial["num_buckets_old"], 
					removed_buckets, 
					trial["num_buckets_new"], 
					f"{trial['old_build_time_ms']:.2f}", 
					f"{merge_time_ms:.2f}",
					f"{merge_throughput:.2f}"
				])

def main(argv):
	input_file = ''
	try:
		opts, args = getopt.getopt(argv, "hvi:m:d:", ["help", "verbose", "ifile=", "mode=", "device="])
	except getopt.GetOptError:
		print("bencher.py -i <inputfile> -m <experiment mode> -d <device index> -v")
		sys.exit(2)
	
	for opt, arg in opts:
		if opt == '-h':
			print("===============================================================================================")
			print("-i/--ifile: 	\t\t Input file (optional)")
			print("-m/--mode: 	\t\t Experiment mode:")
			print("\t\t\t\t\t 0: singleton experiment")
			print("\t\t\t\t\t 1: load factor experiment")
			print("\t\t\t\t\t 2: variable sized table experiment")
			print("\t\t\t\t\t 3: concurrent experiment")
			print("\t\t\t\t\t 4: rehash experiment")
			print("\t\t\t\t\t 5: merge experiment")
			print("-v/--verbose")
			print("===============================================================================================")
			sys.exit()
		else:
			if opt in ("-i", "--ifile"):
				input_file = arg
				print("input file: " + input_file)
			if opt in ("-m", "--mode"):
				mode = int(arg)
			if opt in ("-d", "--device"):
				device_idx = int(arg)
			if opt in ("-v", "--verbose"):
				verbose = True
			else:
				verbose =  False
	
	# if the input file is not given, proper experiments should be run first
	if not input_file:		
		# == creating a folder to store results
		out_directory = "../build/bench_result/"
		if (not os.path.isdir(out_directory)):
			os.mkdir(out_directory)

		# == running benchmark files
		bin_file = "../build/bin/benchmark"
		if(not os.path.exists(bin_file)):
			raise Exception("binary file " + bin_file + " not found!")

		# creating a unique name for the file
		cur_time_list = str(datetime.datetime.now()).split()
		out_file_name = "out"
		for s in cur_time_list:
			out_file_name += ("_" + s)

		out_file_dest = out_directory + out_file_name + ".json"
		input_file = out_file_dest # input file for the next step
		print("intermediate results stored at: " + out_file_dest)

		print("mode = %d" % mode)
		if mode == 0:
			args = (bin_file, "-mode", str(mode), 
				"-num_key", str(2**20),
				"-expected_chain", str(0.6),
				"-device", str(device_idx),
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 1:
			args = (bin_file, 
				"-mode", str(mode),
				"-num_keys", str(2**22),
				"-quary_ratio", str(1.0),
				"-device", str(device_idx),
				"-lf_bulk_step", str(1.0),
				"-lf_bulk_num_sample", str(20), 
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 2:
				args = (bin_file, "-mode", str(mode), 
				"-nStart", str(22), 
				"-nEnd", str(27), 
				"-expected_chain", str(0.9),
				"-query_ratio", str(1.0),
				"-device", str(device_idx),
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 3:
			nBatchSizes = [18, 19, 20, 21, 22, 23, 24]
			csv_out = os.path.join(RESULTS_DIR, "concurrent_experiment.csv")
			# removing the old csv file if exists
			if os.path.exists(csv_out):
				os.remove(csv_out)
			
			for batchSize in nBatchSizes:
				cur_time_list = str(datetime.datetime.now()).split()
				out_file_name = f"out_concurrent_{batchSize}"
				for s in cur_time_list:
					out_file_name += ("_" + s)
				out_file_dest = out_directory + out_file_name + ".json"
				print("intermediate results stored at: " + out_file_dest)

				args = (
					bin_file,
					"-mode", str(mode),
					"-nStart", str(batchSize),
					"-nEnd", str(batchSize + 2),
					"-num_batch", str(4),
					"-init_batch", str(3),
					"-insert_ratio", str(0.5),
					"-delete_ratio", str(0.1),
					"-search_exist_ratio", str(0.4),
					"-lf_conc_step", str(1.0),
					"-lf_conc_num_sample", str(10),
					"-device", str(device_idx),
					"-filename", out_file_dest,
					"-verbose", "1" if verbose else "0"
				)
				print(" === Started benchmarking for batch size 2^%d ... " % batchSize)
				popen = subprocess.Popen(args, stdout = subprocess.PIPE)
				popen.wait()
				if verbose:
					output = popen.stdout.read()
					print(output)
				print(" === Done!")

				analyze_concurrent_experiment(out_file_dest, csv_out)
			print("All concurrent experiments done! Final results stored at: " + csv_out)
			return
		elif mode == 4:
			buckets_to_add = [260000, 522000, 1044000, 2089000]
			csv_out = os.path.join(RESULTS_DIR, "rehash_experiment.csv")
			if os.path.exists(csv_out):
				os.remove(csv_out)

			for added_buckets in buckets_to_add:
				cur_time_list = str(datetime.datetime.now()).split()
				out_file_name = f"out_rehash_{added_buckets}"
				for s in cur_time_list:
					out_file_name += ("_" + s)
				out_file_dest = out_directory + out_file_name + ".json"
				
				args = (bin_file, "-mode", str(mode),
					"-num_key", str(2**20), # Default 1M keys
					"-added_buckets", str(added_buckets),
					"-device", str(device_idx),
					"-filename", out_file_dest,
					"-verbose", "1" if verbose else "0")

				print(f" === Started benchmarking for added_buckets={added_buckets} ... ")
				popen = subprocess.Popen(args, stdout = subprocess.PIPE)
				popen.wait()
				if verbose:
					output = popen.stdout.read()
					print(output)
				print(" === Done!")
				
				analyze_rehash_experiment(out_file_dest, csv_out)
			
			print("All rehash experiments done! Final results stored at: " + csv_out)
			return
		elif mode == 5:
			buckets_to_remove = [260000, 522000, 1044000, 2089000]
			csv_out = os.path.join(RESULTS_DIR, "merge_experiment.csv")
			if os.path.exists(csv_out):
				os.remove(csv_out)

			for removed in buckets_to_remove:
				cur_time_list = str(datetime.datetime.now()).split()
				out_file_name = f"out_merge_{removed}"
				for s in cur_time_list:
					out_file_name += ("_" + s)
				out_file_dest = out_directory + out_file_name + ".json"
				
				args = (bin_file, "-mode", str(mode),
					"-num_key", str(2**20), # Default 1M keys
					"-added_buckets", str(removed), # Treated as buckets to remove in mode 5
					"-device", str(device_idx),
					"-filename", out_file_dest,
					"-verbose", "1" if verbose else "0")

				print(f" === Started benchmarking for removed_buckets={removed} ... ")
				popen = subprocess.Popen(args, stdout = subprocess.PIPE)
				popen.wait()
				if verbose:
					output = popen.stdout.read()
					print(output)
				print(" === Done!")
				
				analyze_merge_experiment(out_file_dest, csv_out)
			
			print("All merge experiments done! Final results stored at: " + csv_out)
			return
		elif mode == 6:
			search_exist_ratios = [1.0, 0.75, 0.5, 0.25, 0.0]
			csv_out = os.path.join(RESULTS_DIR, "query_experiment_varied_exist_ratio.csv")
			if os.path.exists(csv_out):
				os.remove(csv_out)

			for exist_ratio in search_exist_ratios:
				non_exist_ratio = 1.0 - exist_ratio
				cur_time_list = str(datetime.datetime.now()).split()
				out_file_name = f"out_query_{exist_ratio}_{non_exist_ratio}"
				for s in cur_time_list:
					out_file_name += ("_" + s)
				out_file_dest = out_directory + out_file_name + ".json"

				args = (bin_file, "-mode", str(mode),
					"-num_key", str(2**24), # Default 1M keys
					"-search_exist_ratio", str(exist_ratio),
					"-search_non_exist_ratio", str(non_exist_ratio),
					"-device", str(device_idx),
					"-filename", out_file_dest,
					"-verbose", "1" if verbose else "0")

				print(f" === Started benchmarking for exist_ratio={exist_ratio}, non_exist_ratio={non_exist_ratio} ... ")
				popen = subprocess.Popen(args, stdout = subprocess.PIPE)
				popen.wait()
				if verbose:
					output = popen.stdout.read().decode("utf-8")
					print(output)
				print(" === Done!")

				analyze_query_experiment(out_file_dest, csv_out)
			
			print("All query experiments done! Final results stored at: " + csv_out)
			return

		print(" === Started benchmarking ... ")

		popen = subprocess.Popen(args, stdout = subprocess.PIPE)
		popen.wait()

		if verbose:
			output = popen.stdout.read()
			print(output)
		print(" === Done!")
	elif not os.path.exists(input_file):
		raise Exception("Input file " + input_file + " does not exist!")

	csv_results = os.path.abspath(RESULTS_DIR)
	if( not os.path.isdir(csv_results)):
		os.mkdir(csv_results)

	# reading the json files:
	if mode == 0:
		analyze_singleton_experiment(input_file)
	elif mode == 1:
		analyze_load_factor_experiment(input_file)
	elif mode == 2:
		analyze_table_size_experiment(input_file, os.path.join(csv_results, "table_size_experiment.csv"))
	elif mode == 3:
		analyze_concurrent_experiment(input_file, os.path.join(csv_results, "concurrent_experiment.csv"))	
	elif mode == 4:
		# For single file analysis if passed manually
		analyze_rehash_experiment(input_file)
	elif mode == 5:
		analyze_merge_experiment(input_file)
	elif mode == 6:
		analyze_query_experiment(input_file, csv_results)
	else:
		print("Invalid mode entered")
		sys.exit(2)

if __name__ == "__main__":
	main(sys.argv[1:])