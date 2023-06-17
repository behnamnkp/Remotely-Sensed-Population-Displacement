# Download and install wget and follow the instructions below:
# 1) Download from here: https://eternallybored.org/misc/wget/
# 2) Move it to C:\Windows\System32 (admin privileges will be required)
# 3) Click the Windows Menu > Run
# 4) Type cmd and hit Enter
# 5) In the Windows Command Prompt, type wget -h to verify the binary is being executed successfully
# 6) Go to your LAADS DAAC profile and generate a token. Something similar to this:
# YmVobmFtbmtwONNtNXBhM0JoY25aQWRXNWpZeTVsWkhVPToxNjAyODY3MDk4OmJhODYzZTJlMDI4ZTNiZTRhZTliMjQyODljZTZjZWMyMjQ1OWM0MDk
# 7) search the data in LAADS DAAC website for the area and time of interest. Select all results and download the list
# as a csv file (top left of the page)
# 8) move the csv file to C:\wgetdown
# 9) change the column names to [fileId,	route, size]
# 10) create a "data folder in the same directory as the .csv file"
# 11) change the paths and token in the code below
# 12) Run this code
# 13) open a cmd and write cd C:\wgetdown
# 14) copy the entire output from 12 and paste in cmd
# 15) It must start downloading the files

import pandas as pd
from os import listdir
from os.path import isfile, join

npplist = pd.read_csv(r'C:\wgetdown\MosulVNP2\LAADS_query.2021-02-15T22_29.csv') # path to the .csv
mypath = r'C:\wgetdown\MosulVNP2\raw2012' # path to the output directory (data)
onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f))) and '.h5' in f]

# replace the token value with that of yours
for item in npplist.route:
    if item[-44:] not in onlyfiles:
        print("wget -e robots=off -m -np -nd -r .html,.tmp -nH --cut-dirs=3 " + \
                str("https://ladsweb.modaps.eosdis.nasa.gov" + \
                item) + \
                " --header " + \
                str('"Authorization: Bearer YmVobmFtbmtwOlltNXBhM0JoY25aQWRXNWpZeTVsWkhVPToxNjEzNDI4NDI0OjVjOTE2MjY3NWViZWY3Mzc0M2FjYTczNjk0MDRhODhmMTA2ODUyOTg"') + \
                ' -P ' + \
                r'C:\wgetdown\MosulVNP2\raw2012')# output directory