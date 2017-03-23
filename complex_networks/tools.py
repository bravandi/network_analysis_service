import pdb
from datetime import datetime
import os
import subprocess


def convert_string_datetime(input):
    input = input.strip()
    if input == '':
        return None
    else:
        return datetime.strptime(input.split('.')[0], "%Y-%m-%d %H:%M:%S")


def get_time_difference(start_time, end_time=None):
    if isinstance(start_time, str):
        start_time = convert_string_datetime(start_time)

    # if isinstance(end_time, strbasestring): # strbasestring probably not in python 3.5
    if isinstance(end_time, str):
        end_time = convert_string_datetime(end_time)

    if end_time is None:
        end_time = datetime.now()

    difference = (end_time - start_time)
    return difference.total_seconds()


def run_command(parameters, debug=False, no_pipe=False):
    if no_pipe is True:
        p = subprocess.Popen(parameters)

        return p

    else:

        p = subprocess.Popen(parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = p.communicate()

        if debug:
            print("\nRUN_COMMAND:\n" + str(parameters) + "\nSTDOUT -->" + out + "\nSTDERR --> " + err)

        return out, err, p


def run_command2(command, get_out=False, debug=False, no_pipe=False):
    task = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    out = ''

    if get_out is True:
        out = task.stdout.read()
        assert task.wait() == 0

    if debug:
        print("\nRUN_COMMAND2:\n" + command + "\nOUT -->" + out)

    return out


def read_file(file_name, hard_path="~/cinder/cinder/MLScheduler/"):
    # os.path.realpath(
    with open(os.path.expanduser(hard_path + file_name), 'r') as myfile:
        data = myfile.read()

    myfile.close()
    return data


def log(
        message,
        experiment_id=0,  # the database will insert the latest experiment id
        volume_cinder_id='',
        type='',
        app='MLScheduler',
        code='',
        file_name='',
        function_name='',
        exception='',
        create_time=None,
        insert_db=True):
    return
    #
    exception_message = str(exception)

    if create_time is None:
        create_time = datetime.now()

    args = (
        experiment_id,
        volume_cinder_id,
        app,
        type,
        code,
        file_name,
        function_name,
        message,
        exception_message,
        create_time,
        -1
    )

    if exception != '':
        exception = "\n   ERR: " + str(exception)

    msg = "\n {%s} <%s>-%s [%s - %s] %s. [%s] %s\n" \
          % (
              app, type, code, function_name, file_name, message, create_time.strftime("%Y-%m-%d %H:%M:%S"),
              str(exception))

    print (msg)

    if insert_db is False:
        return msg

    try:
        conn = database.__create_connection_for_insert_delete()
        cursor = conn.cursor()
        output = cursor.callproc("insert_log", args)
        conn.commit()

        # output = list(output)
        # insert_id = output[len(output) - 1]
        #
        # print insert_id

    except Exception as err:
        pdb.set_trace()
        raise Exception("ERROR in LOGGING. ARGS -->%s\n\nERR-->%s" % (args, str(err)))

    finally:

        cursor.close()
        conn.close()

    return msg


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_path_for_tenant(var=""):
    if var.startswith("~"):
        var = var[1:]

    if var.startswith("/"):
        var = var[1:]

    return "/home/centos/" + var


if __name__ == '__main__':
    # out = run_command2("ls /root > ls.out")
    # print ("ou: " + out)



    pass
