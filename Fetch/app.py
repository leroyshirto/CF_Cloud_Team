import oisp
from logging import basicConfig, getLogger
import pickle
import numpy
import pdb
import datetime
# from itertools import groupby
from minio import Minio
from minio.error import ResponseError
import os


class OispDataStreamer:

    def __init__(self):
        self.__client__ = oisp.Client(
            api_root="http://cloudfest.streammyiot.com/v1/api")
        self.__client_username__ = "edge@example.com"
        self.__client_password__ = "password"
        self.auth_client()
        self.__accounts__ = self.get_accounts()
        self.__account__ = self.__accounts__[0]
        self.set_account_to_use(self.__account__)
        self.__devices__ = self.get_devices()
        self.file = 'oisp_query_data_{}'.format(datetime.datetime.now())
        self.__minioClient__ = Minio('212.227.4.254:9000',
                                     access_key='AKIAIOSFODNN7EXAMPLE',
                                     secret_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
                                     secure=False)

    def create_account(self, accountname):
        """

        :param accountname: a string for the account name
        :return:
        """

        self.__account__ = self.__client__.create_account(accountname)
        self.auth_client()

    def auth_client(self):
        """

        :return: This will auth the client to the server
        """
        self.__client__.auth(self.__client_username__,
                             self.__client_password__)

    def get_accounts(self):
        """

        :return: an array with objects of all accounts
        """
        for account in self.__client__.get_accounts():
            getLogger(__name__).info(account)
            # self.__accounts__.append(account)
        return self.__client__.get_accounts()

    def set_account_to_use(self, account):
        """

        :return:
        """
        self.__account__ = account
        getLogger(__name__).info("Selected {}".format(self.__account__))

    def create_device(self, deviceid, devicename):
        """

        :param deviceid: a unique string to identify the device (hint use the MAC)
        :param devicename: a string with your favourite name
        :return: None
        """
        self.__account__.create_device(deviceid, devicename)
        getLogger(__name__).info(
            self.__account__.create_device(deviceid, devicename))

    def get_devices(self):
        """

        :return: a list of all devices for on account
        """
        getLogger(__name__).info(self.__account__.get_devices())
        # for device in self.__account__.get_devices():
        #    self.__devices__.append(device)
        return self.__account__.get_devices()

    def activate_device(self, device):
        """

        :param device: expect an device.Device object
        :return:
        """
        device.activate()

    def get_account_data(self):
        """

        :return: returns all data from one account
        """
        query = oisp.DataQuery()
        response = self.__account__.search_data(query)
        data_values = [sample.value for sample in response.samples]
        getLogger(__name__).info("DATA FROM QUERY: {}".format(data_values))
        return data_values

    def get_device_data(self, device_id, cid):
        query = oisp.DataQuery(device_ids=[device_id], component_ids=[cid],
                               from_=oisp.utils.timestamp_in_ms() - 1000 * 60 * 1)
        response = self.__account__.search_data(query)
        data_values = [sample.value for sample in response.samples]
        getLogger(__name__).info("DATA FROM QUERY: {}".format(data_values))
        return data_values

    def upload_file(self, file_name):
        try:
            with open(file_name, 'br') as file_data:
                file_stat = os.stat(file_name)
                self.__minioClient__.put_object(
                    "input", file_name, file_data, file_stat.st_size)
        except ResponseError as err:
            print(err)


def main():
    oisp = OispDataStreamer()
    data = oisp.get_device_data('cfh_device_id_edge_1',
                                '969ca3c4-bf08-49ca-9e70-aae40e5b7fc6')
    npy_data = pickle.loads(data[-1])[-1]
    numpy.save(oisp.file, npy_data)
    getLogger(__name__).info("Numpy Array: {}".format(npy_data))
    oisp.upload_file(file_name=f"{oisp.file}.npy")
    with open('/tmp/output_file', 'w') as file:
        file.write("s3://input/{}.npy".format(oisp.file))
    # pdb.set_trace()


if __name__ == '__main__':
    basicConfig(level="ERROR", format="%(asctime)s %(name)s %(message)s")
    main()
