import synapseclient
syn = synapseclient.login("catalin.iscruleasa", "21022001Cc")
dl_list_file_entities = syn.get_download_list()

# Specify the local directory for downloads
download_dir = "/path/to/download/directory"

# Download the files
for item in dl_list_file_entities:
    syn.download(item["entity"], downloadLocation=download_dir)
