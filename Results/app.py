import requests
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Component executing inference operation')
    parser.add_argument('--results_json', type=str, required=True,
                        help='json object to submit')
    args = parser.parse_args()
    print(args)

    results_data = json.loads(args.results_json)

    submission = {
        'data': results_data
    }

    url = "http://results.cluster2.cf2019.42m.co.uk/api/v1/results"

    headers = {
        'Accept': "application/json",
        'Content-Type': "application/json",
        'Authorization': "Bearer 1176e6ff-dff9-41e2-b92d-13f45cf842cb",
    }

    response = requests.request("POST", url, data=results_data, headers=headers)
