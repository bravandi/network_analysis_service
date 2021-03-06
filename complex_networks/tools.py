import pdb
from datetime import datetime
import os
import subprocess
import networkx as nx
import snap
import constants
from Tkinter import Tk
import pylab
import matplotlib


def color_generator(NUM_COLORS=22):
    cm = pylab.get_cmap('gist_rainbow')
    # cmap = pylab.get_cmap('seismic', 5)
    #
    # for i in range(cmap.N):
    #     rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
    #     print(matplotlib.colors.rgb2hex(rgb))


    # or if you really want a generator:
    cm = pylab.get_cmap('gist_rainbow')
    cgen = (
        matplotlib.colors.rgb2hex(cm(1. * i / NUM_COLORS)) for i in range(NUM_COLORS))

    return cgen


def path_split(path, return_index=None):
    spl = path.split(os.sep)

    if return_index is None:
        return spl

    return (spl, spl[return_index],)


def relative_path(path):
    if os.path.isfile(path):
        return path

    return os.path.abspath(constants.path_work + path)


def absolute_path(path):
    return os.path.abspath(path)


def networkx_draw(G, path, label='', positions=None):
    pos = nx.nx_pydot.graphviz_layout(G)

    if not os.path.exists(os.path.dirname(path + '.dot')):
        os.makedirs(os.path.dirname(path + '.dot'))

    path = absolute_path(path)

    G.graph['graph'] = {
        'splines': True,
        'ranksep': 3,
        'nodesep': 0.8,
        '#ratio': 1,
        'label': label
    }

    if positions is not None:
        nx.set_node_attributes(G, 'pos', positions)

    nx.nx_pydot.write_dot(G, path + '.dot')

    if positions is None:
    # run_command("dot -Tpng %s > %s" % (path + '.dot', path))
        run_command("dot -Tjpg %s -o %s" % (path + '.dot', path), no_pipe=True)
    else:
        run_command("dot -Kfdp -n -Tjpg %s -o %s" % (path + '.dot', path), no_pipe=True)

    return path


def snap_get_graph_nodes(snap_g):
    return [n.GetId() for n in snap_g.Nodes()]


def snap_get_graph_edges(snap_g):
    return [(e.GetSrcNId(), e.GetDstNId()) for e in snap_g.Edges()]


def create_path_if_not_exists(path):
    path = os.path.realpath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def snap_draw(G, path, name="graph 1"):
    path = os.path.realpath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    snap.DrawGViz(G, snap.gvlDot, path + ".png", name, True)

    return path + ".png"


def convert_string_datetime(input):
    input = input.strip()
    if input == '':
        return None
    else:
        return datetime.strptime(input.split('.')[0], "%Y-%m-%d %H:%M:%S")


def get_time_difference_seconds(start_time, end_time=None):
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


def clipboard_copy(str):
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(str)
    r.destroy()


def directory_files(from_path):
    for (dirpath, dirnames, filenames) in os.walk(from_path):
        if dirpath == from_path:
            return filenames

    return None


if __name__ == '__main__':
    # out = run_command2("ls /root > ls.out")
    # print ("ou: " + out)



    pass
