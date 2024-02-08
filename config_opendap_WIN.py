#!/usr/bin/env python3
# Script to create configuration files .dodsrc and _netrc
# for an easy access to the OPeNDAP API

import os
import getpass

# Enter Copernicus credentials
USERNAME = getpass.getpass('Enter your Copernicus username: ')
PASSWORD = getpass.getpass('Enter your Copernicus password: ')

# Create .dodsrc
f = open(os.path.expanduser('~')+"/.dodsrc", "w", encoding="utf-8")
f.write("HTTP.NETRC="+os.path.expanduser('~')+"/_netrc"+'\n')
f.write("HTTP.COOKIEJAR="+os.path.expanduser('~')+"/.cookies")
f.close()

# Create _netrc
f = open(os.path.expanduser('~')+"/_netrc", "w", encoding="utf-8")
f.writelines(["machine my.cmems-du.eu", '\n', "login "+USERNAME, '\n', "password "+PASSWORD, '\n', '\n'])
f.writelines(["machine nrt.cmems-du.eu", '\n', "login "+USERNAME, '\n', "password "+PASSWORD, '\n', '\n'])
f.close()
