#!/usr/bin/env python3
# Script to create configuration files .dodsrc and _netrc (or .netrc)
# for an easy access to the OPeNDAP API

# Import libraries
import getpass
from pathlib import Path
from platform import system
from os.path import exists

HOME = Path.home()
netrc_file = HOME / "_netrc" if system() == "Windows" else HOME / ".netrc"
dodsrc_file = HOME / ".dodsrc"
cookies_file = HOME / ".cookies"
OPeNDAP_SERVERS = ["my.cmems-du.eu", "nrt.cmems-du.eu"]

if not exists(netrc_file):
    username = input("Enter your Copernicus Marine username: ")
    password = getpass.getpass("Enter your Copernicus Marine password: ")

    # Create netrc file
    with open(netrc_file, "a") as file:
        for server in OPeNDAP_SERVERS:
            file.write(f"machine {server}\nlogin {username}\npassword {password}\n\n")
        
if not exists(dodsrc_file):
    # Create dodsrc file
    with open(dodsrc_file, "a") as file:
        file.write(f"HTTP.NETRC={netrc_file}\nHTTP.COOKIEJAR={cookies_file}")