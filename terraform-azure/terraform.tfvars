location = "southindia"            # choose close region (centralindia / southindia)
resource_group_name_prefix = "mc-rg"
admin_username = "azureuser"
ssh_public_key_path = "~/.ssh/id_rsa.pub"   # ensure file exists; generate with ssh-keygen if needed
vm_size = "Standard_B1s"
vm_count = 2
repo_url = "https://github.com/Hariprakash024/Monte-Carlo-Simulation-in-supply-chain-management.git"  # main has src/mc_mpi_cpu.cpp
