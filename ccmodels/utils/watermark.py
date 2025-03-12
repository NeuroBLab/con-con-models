from subprocess import check_output

def get_commit_hash():
    return check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")[:-1]

def add_metadata(d=dict(), extra=""):
    meta = "git commit id "
    meta += get_commit_hash() + "; "
    for key in d:
        meta += f"{key} : {d[key]},"
    
    meta += f"; {extra}"
    return meta 


