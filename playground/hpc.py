from paramiko import SSHClient
from getpass import getpass
import time


class HPC:
    def __init__(self, hostname = 'login1.gbar.dtu.dk') -> None:
        # Connect
        self.client = SSHClient()
        self.client.load_system_host_keys()
        username = input("Username: ")
        pw = getpass("Password: ")
        self.client.connect(hostname, username=username, password=pw)
    

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

    def close(self):
        self.client.close()


if __name__ == "__main__":
    HPC().shell()


# stdin, stdout, stderr = client.exec_command(command)
            # print(f'STDOUT: {stdout.read().decode("utf8")}')
            # print(f'STDERR: {stderr.read().decode("utf8")}')


# Run a command (execute PHP interpreter)
# stdin, stdout, stderr = client.exec_command("; ".join([
#     f"cd /work1/{username}/CoolData",
#     "ls"
# ]))


   




# stdin, stdout, stderr = client.exec_command('source CoolData-cluster-env/bin/activate')
# stdin, stdout, stderr = client.exec_command('echo $PWD')

# print(type(stdin))  # <class 'paramiko.channel.ChannelStdinFile'>
# print(type(stdout))  # <class 'paramiko.channel.ChannelFile'>
# print(type(stderr))  # <class 'paramiko.channel.ChannelStderrFile'>

# Optionally, send data via STDIN, and shutdown when done
# stdin.write('<?php echo "Hello!"; sleep(2); ?>')
# stdin.channel.shutdown_write()

# Print output of command. Will wait for command to finish.
# print(f'STDOUT: {stdout.read().decode("utf8")}')
# print(f'STDERR: {stderr.read().decode("utf8")}')

# Get return code from command (0 is default for success)
# print(f'Return code: {stdout.channel.recv_exit_status()}')

# Because they are file objects, they need to be closed
# stdin.close()
# stdout.close()
# stderr.close()

# Close the client itself

