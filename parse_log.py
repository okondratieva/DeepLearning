import os
import csv
import re

def parse_log(log, output_name):
    _log = open(log, 'r')
    _csv = open(output_name + '.csv', 'w', newline='')
    csv_writer = csv.writer(_csv, delimiter=';')

    statistic = []
    tmp = [0, 0, 0, 0]
    config_done = False
    for line in _log:
        result = re.match('(.{0,20})\: (.*)', line)
        if result and not config_done:
            csv_writer.writerow([result.group(1), result.group(2)])
        result = re.match('.*Train-accuracy=(.*)', line)
        if result:
            config_done = True
            tmp[0] += 1
            tmp[1] = result.group(1)
        result = re.match('.*Validation-accuracy=(.*)', line)
        if result:
            config_done = True
            tmp[2] = result.group(1)
            statistic.append(tmp.copy())
            tmp[1:] = [0, 0, 0]
        result = re.match('.*Time cost=(.*)', line)
        if result:
            config_done = True
            tmp[3] = result.group(1)
    _log.close()
    csv_writer.writerow(['epoch', 'train', 'test', 'time'])

    max_test, max_train = 0, 0
    max_test_epoch, max_train_epoch = 0, 0
    for stat in statistic:
        train = float(stat[1])
        test = float(stat[2])

        if (train > max_train):
            max_train = train
            max_train_epoch = stat[0]
        if (test > max_test):
            max_test = test
            max_test_epoch = stat[0]
        csv_writer.writerow(stat)
    csv_writer.writerow(['max', max_train, max_test])
    csv_writer.writerow(['max epoch', max_train_epoch, max_test_epoch])

    _csv.close()

parse_log('test9.log', 'test9')