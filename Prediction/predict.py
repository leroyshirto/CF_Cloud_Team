from openvino.inference_engine import IENetwork, IEPlugin
import argparse
import numpy as np
from urllib.parse import urlparse
import classes
import datetime
from shutil import copy
import os
import json
from minio import Minio
from minio.error import ResponseError

minioClient = Minio('212.227.4.254:9000',
                    access_key='AKIAIOSFODNN7EXAMPLE',
                    secret_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
                    secure=False)

''' source_path = "s3://bucke_name/filename[file_type] '''


def get_local_file(source_path):
    parsed_path = urlparse(source_path)
    if parsed_path.scheme == "s3":
        bucket_name = parsed_path.netloc
        file_name = parsed_path.path[1:]
        print("Extracted Bucket Name =" +
              bucket_name + " File Name =" + file_name)

        try:
            data = minioClient.get_object(bucket_name, file_name)
            with open(file_name, 'wb') as file_data:
                for d in data.stream(32*1024):
                    file_data.write(d)
        except ResponseError as err:
            print(err)
    elif parsed_path.scheme == "":
        # in case of local path just pass the input argument
        if os.path.isfile(source_path):
            file_name = source_path
        else:
            print("file " + source_path + "is not accessible")
            file_name = ""
    return file_name


''' output_path = "s3://bucke_name/filename[file_type] '''


def upload_file(output_bucket, file_name):
    parsed_path = urlparse(output_bucket)
    if parsed_path.scheme == "s3":
        output_bucket = parsed_path.netloc
        print("Using Bucket Name =" +
              output_bucket + " File Name =" + file_name)

        try:
            with open(file_name, 'rb') as file_data:
                file_stat = os.stat(file_name)
                minioClient.put_object(output_bucket, file_name,
                                       file_data, file_stat.st_size)
        except ResponseError as err:
            print(err)
    elif parsed_path.scheme == "":
        if output_bucket != ".":
            copy(file_name, output_bucket)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Component executing inference operation')
    parser.add_argument('--model_bin', type=str, required=True,
                        help='GCS or local path to model weights file (.bin)')
    parser.add_argument('--model_xml', type=str, required=True,
                        help='GCS or local path to model graph (.xml)')
    parser.add_argument('--input_numpy_file', type=str, required=True,  # Provided by Fetch Script
                        help='GCS or local path to input dataset numpy file')
    parser.add_argument('--label_numpy_file', type=str, required=True,  # Provided by Fetch Script
                        help='GCS or local path to numpy file with labels')
    parser.add_argument('--output_bucket', type=str, required=True,
                        help='S3 Compatible Bucket or local path to results upload folder')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size to be used for inference')
    parser.add_argument('--scale_div', type=float, default=1,
                        help='scale the np input by division of by the value')
    parser.add_argument('--scale_sub', type=float, default=128,
                        help='scale the np input by substraction of the value')
    args = parser.parse_args()
    print(args)

    device = "CPU"
    plugin_dir = None

    model_xml = get_local_file(args.model_xml)
    print("model xml", model_xml)
    if model_xml == "":
        exit(1)
    model_bin = get_local_file(args.model_bin)
    print("model bin", model_bin)
    if model_bin == "":
        exit(1)
    input_numpy_file = get_local_file(args.input_numpy_file)
    print("input_numpy_file", input_numpy_file)
    if input_numpy_file == "":
        exit(1)

    label_numpy_file = get_local_file(args.label_numpy_file)
    print("label_numpy_file", label_numpy_file)
    if label_numpy_file == "":
        exit(1)

    cpu_extension = "/usr/local/lib/libcpu_extension.so"

    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)

    print("inference engine:", model_xml, model_bin, device)

    # Read IR
    print("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)
    batch_size = args.batch_size
    net.batch_size = batch_size
    print("Model loaded. Batch size", batch_size)

    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    print(output_blob)

    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=1)

    print("Loading input numpy")
    imgs = np.load(input_numpy_file, mmap_mode='r', allow_pickle=False)
    imgs = (imgs / args.scale_div) - args.scale_div
    lbs = np.load(label_numpy_file, mmap_mode='r', allow_pickle=False)

    print("Loaded input data", imgs.shape, imgs.dtype,
          "Min value:", np.min(imgs), "Max value", np.max(imgs))

    combined_results = {}  # dictionary storing results for all model outputs
    processing_times = np.zeros((0), int)
    matched_count = 0
    total_executed = 0

    for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
        img = imgs[x:(x + batch_size)]
        lb = lbs[x:(x + batch_size)]
        start_time = datetime.datetime.now()
        results = exec_net.infer(inputs={input_blob: img})
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("Inference duration:", duration, "ms")
        processing_times = np.append(
            processing_times, np.array([int(duration)]))
        output = list(results.keys())[0]  # check only one output
        nu = results[output]
        for i in range(nu.shape[0]):
            single_result = nu[[i], ...]
            ma = np.argmax(single_result)
            total_executed += 1
            if ma == lb[i]:
                matched_count += 1
                mark_message = "; Correct match."
            else:
                mark_message = "; Incorrect match. Should be {} {}".format(
                    lb[i], classes.imagenet_classes[lb[i]])
            print("\t", i, classes.imagenet_classes[ma], ma, mark_message)
        if output in combined_results:
            combined_results[output] = np.append(combined_results[output],
                                                 results[output], 0)
        else:
            combined_results[output] = results[output]

    filename = output.replace("/", "_") + ".npy"
    np.save(filename, combined_results[output])
    upload_file(filename, args.output_folder)
    print("Inference results uploaded to", filename)
    print('Classification accuracy: {:.2f}'.format(
        100*matched_count/total_executed))
    print('Average time: {:.2f} ms; average speed: {:.2f} fps'.format(round(np.average(
        processing_times), 2), round(1000 * batch_size / np.average(processing_times), 2)))

    accuracy = matched_count/total_executed
    latency = np.average(processing_times)
    metrics = {'metrics': [{'name': 'accuracy-score', 'numberValue':  accuracy, 'format': "PERCENTAGE"},
                           {'name': 'latency', 'numberValue':  latency, 'format': "RAW"}]}

    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
