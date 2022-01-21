from paramiko import SSHClient, AutoAddPolicy
from getpass import getpass
import time
import sys, os
from autovc.utils import pformat
import pandas as pd

class HPC:
    def __init__(self, hostname = 'login1.gbar.dtu.dk') -> None:
        # get credentials 
        self.hostname = hostname
        self.__username = input("Username: ")
        self.__pw = getpass("Password: ")

        # Connect
        self.client = SSHClient()
        self.client.set_missing_host_key_policy(AutoAddPolicy())
        self.client.load_system_host_keys()
        self.client.connect(hostname, username=self.__username, password=self.__pw, port = '22')#, disabled_algorithms=dict(pubkeys=["rsa-sha2-512"]))
    

    def shell(self):
        # open channel
        channel = self.client.get_transport().open_session()
        channel.get_pty()
        channel.invoke_shell()

        # read welcome message
        time.sleep(1)
        while not channel.recv_ready():
            print("Working...")
            time.sleep(2)
        print(channel.recv(4096).decode("utf-8"))

        # interactive commands
        while True:
            # get command
            command = input('$ ')

            # exit
            if command == 'exit':
                break
            
            # send command
            channel.send(command + "\n")
            time.sleep(.1)

            # read output
            while not channel.recv_ready():
                print("Working...")
                time.sleep(2)
            out = channel.recv(1024).decode("utf-8")
            out = "\n".join(out.split("\n")[1:-2])
            print(out)
        
        # close connection
        self.close()

    def send_data(self, data_path, receive_folder, gbar = True):
        if gbar and "transfer.gbar.dtu.dk" not in receive_folder:
            receive_folder = self.__username + "@transfer.gbar.dtu.dk:" + receive_folder
  
        if "win" in sys.platform:
            command = f'pscp -pw "{self.__pw}" -r {data_path} {receive_folder}'
        else:
            command = f'rsync -r {data_path} {receive_folder}'

        os.system(command)

    def receive_data(self, data_path, receive_folder, gbar = True):
        os.makedirs(receive_folder, exist_ok=True)

        if gbar and "transfer.gbar.dtu.dk" not in data_path:
            data_path = self.__username + "@transfer.gbar.dtu.dk:" + data_path

        if "win" in sys.platform:
            command = f'pscp -pw "{self.__pw}" -scp {data_path} {receive_folder}'
        else:
            command = f'rsync -r {data_path} {receive_folder}'

        os.system(command)
        

    def close(self):
        self.client.close()

def create_submit(jobname: str, project:str, *scripts, **kwargs):
    """
    Creates a jobscript that can be submitted to a cluster. Everything will be saved in the 'logs/' directory

    Parameters
    ----------
    jobname:
        name of the job, if using wandb, this should be set as the wandb run name
    project:
        name of the project folder to save the job script and output files to
    *scripts:
        a str with code to run at the end of the batch job, will typically be something like `'python script.py -flag value'`
    **kwargs:
        parameters to pass to the cluster, this includes `use_gpu`, `queue_name`, `n_cores`, `walltime`, `system_memory`, `notifications`
    
    Returns
    -------
    filename:
        the file name of the created job script

    Output
    ------
    The script will be saved to the folder 'logs/<project>/cluster_submits/' and the output logs will be saved to 'logs/<project>/hpc/'
    """
    # assert script is run from correct path
    cur_dir = os.path.split(os.getcwd())[-1]
    assert cur_dir == "AutoVC", f"current directory must be AutoVC, not {cur_dir}"
    
    # set create folder for experiment
    # if project is None:
    #     project = ""
    # else:
    folder = f"logs/{project}/cluster_submits/"
    os.makedirs(folder, exist_ok=True)
    
    # set filename
    filename = folder + f"submit_{jobname}.sh"

    # action if file already exists
    if os.path.exists(filename):
        if kwargs.get("overwrite", False):
            print(pformat.YELLOW, f"The file {filename} has been overwritten", pformat.END)
            os.remove(filename)
        else:
            raise FileExistsError(f"File '{filename}' already exists, use overwrite=True or specify another name")

    with open(filename, 'a') as output, open('scripts/templates/batchjob.sh', 'r') as input:
        content = input.read()

        content = content.replace('mkdir -p "logs/hpc"', f'mkdir -p "logs/{project}/hpc"')
        
        # choose between gpu and cpu
        use_gpu = kwargs.get("use_gpu", True)
        queue_name = kwargs.get("queue_name", "gpuv100" if use_gpu else "hpc")
        content = content.replace("queue_name", queue_name)
        
        # set job name
        content = content.replace("#BSUB -J TemplateJob", f"#BSUB -J {jobname}")

        # choose number of cores
        n_cores = kwargs.get("n_cores", 1)
        if use_gpu:
            content = content.replace(
                "#BSUB -n 1", 
                f"#BSUB -n {n_cores}\n" +
                f"### -- Select the resources: {n_cores} gpu in exclusive process mode --\n" +
                f'#BSUB -gpu "num={n_cores}:mode=exclusive_process"')
            # if n_cores > 1:
            #     content = content.replace()
        else:
            content = content.replace("#BSUB -n 1", f"#BSUB -n {n_cores}")
        
        if n_cores > 1:
            content = content.replace(f"#BSUB -n {n_cores}",
            f"#BSUB -n {n_cores}\n" +
            '### -- specify that the cores must be on the same host -- \n' +
            '#BSUB -R "span[hosts=1]"\n' + 
            '### -- specify that we need 2GB of memory per core/slot -- \n' +
            '#BSUB -R "rusage[mem=2GB]"'
            )
        
        # set wall time
        walltime = kwargs.get("walltime", "8:00")
        content = content.replace("#BSUB -W 8:00", f"#BSUB -W {walltime}")

        # set system memory
        system_memory = kwargs.get("system_memory", 5)
        content = content.replace("### -- request 5GB of system-memory --", f"### -- request {system_memory}GB of system-memory --")
        content = content.replace('#BSUB -R "rusage[mem=5GB]"', f'#BSUB -R "rusage[mem={system_memory}GB]"')
            
        # notifications
        
        notifications = kwargs.get("notifications", True)
        if not notifications:
            content = content.replace(
                "### -- set the email address --\n" + 
                "##BSUB -u studentnumber@student.dtu.dk\n" + 
                "### -- send notification at start --\n"+
                "#BSUB -B\n" +
                "### -- send notification at completion-- \n" +
                "#BSUB -N \n",
                ""
            )
        else:
            content = content.replace("studentnumber", os.getlogin())
        
        # set name of output and error file
        content = content.replace("status_file_name", f"logs/{project}/hpc/{jobname}")

        # add scripts to run
        content += "\n".join(scripts)

        # write content to file
        output.write(content)
        
    return filename



if __name__ == "__main__":
    # HPC().shell()
    hpc = HPC()
    # hpc.receive_data("/work1/s183920/AutoVC/results/SMK_material/*", "results/SMK_material")
    hpc.receive_data("/work1/s183920/AutoVC/results/HY1/*", "results/HY1")
    # hpc.send_data("data/long_hilde/Hilde.wav", "/work1/s183920/AutoVC/data/SMK_train/")

    # file = hpc.client.open_sftp()
    # file.get("s183920@transfer.gbar.dtu.dk:Downloads/args.py", "playground/")
    # file.close()

    # hpc.close()


# stdin, stdout, stderr = self.client.exec_command(command)
#         # stdin, stdout, stderr = self.client.exec_command('echo $PWD')

#         print(type(stdin))  # <class 'paramiko.channel.ChannelStdinFile'>
#         print(type(stdout))  # <class 'paramiko.channel.ChannelFile'>
#         print(type(stderr))  # <class 'paramiko.channel.ChannelStderrFile'>

#         # Optionally, send data via STDIN, and shutdown when done
#         # stdin.write('<?php echo "Hello!"; sleep(2); ?>')
#         # stdin.channel.shutdown_write()

#         # Print output of command. Will wait for command to finish.
#         print(f'STDOUT: {stdout.read().decode("utf8")}')
#         print(f'STDERR: {stderr.read().decode("utf8")}')

#         # Get return code from command (0 is default for success)
#         print(f'Return code: {stdout.channel.recv_exit_status()}')

#         # Because they are file objects, they need to be closed
#         stdin.close()
#         stdout.close()
#         stderr.close()

