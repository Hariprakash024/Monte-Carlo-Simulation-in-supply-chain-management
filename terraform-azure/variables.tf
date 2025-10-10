variable "location" { type=string, default="eastus", description="Azure region" }
variable "resource_group_name_prefix" { type=string, default="mc-rg" }
variable "admin_username" { type=string, default="azureuser" }
variable "ssh_public_key_path" { type=string, default="~/.ssh/id_rsa.pub" }
variable "vm_size" { type=string, default="Standard_B1s" }
variable "vm_count" { type=number, default=2 }
variable "repo_url" { type=string, default="https://github.com/your-user/your-repo.git" }  # CHANGE THIS
