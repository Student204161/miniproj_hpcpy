#Let <username> be an input to script
username=$1
# Download the file from the HPC server to your local machine
mkdir -p data
ssh $username@login.hpc.dtu.dk '
  cd /dtu/projects/02613_2025/data/modified_swiss_dwellings &&
  ls | head -n 100 | tar -czf - -T - 
' > first_100_files.tar.gz
#extract the tar.gz file to folder data/
tar -xzf first_100_files.tar.gz -C data/
# Remove the tar.gz file
rm first_100_files.tar.gz