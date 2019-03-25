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
from inference_engine import InferenceEngine

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
                        help='S3 Compatible Bucket or local path to input dataset numpy file')
    parser.add_argument('--output_bucket', type=str, required=True,
                        help='S3 Compatible Bucket or local path to results upload folder')
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

    print("inference engine:", model_xml, model_bin, device)
    engine = InferenceEngine(
        model_bin=model_bin, model_xml=model_xml, device=device)

    # Load NPY
    raw_frame = np.load(input_numpy_file)

    # Decode RAW JPEG Frame into frame usable by InferenceEngine
    frame = cv2.imdecode(raw_frame, 1)

    emotionsVec = ["neutral", "happy", "sad", "surprise", "anger"]

    index = None

    if infer_engine.submit_request(frame, True):
        result = infer_engine.fetch_result()

        if result.any():
            result = result[0]
            if len(result) == len(emotionsVec):
                prob_value = 0.0
                for i in range(len(result)):
                    if result[i][0][0] > prob_value:
                        prob_value = result[i][0][0]
                        index = i
    if index is not None:
        # Store JSON
        return (emotionsVec[index])


if __name__ == "__main__":
    main()
