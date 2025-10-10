# Azure Terraform CPU Cluster (2 VMs)

## Prereqs
- Azure CLI: az login (select Azure for Students)
- SSH key at ~/.ssh/id_rsa.pub (or edit ssh_public_key_path)
- Terraform >= 1.3

## Deploy
cd terraform-azure
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars -auto-approve

## Connect and run MPI
terraform output private_ips
terraform output ssh_commands
ssh azureuser@<PUBLIC_IP_0>

# On VM0:
cd ~/hpc-montecarlo/src
mpicxx -O3 -std=c++17 mc_mpi_cpu.cpp -o mc_mpi_cpu  # if needed
cat > hosts <<EOF
<PRIVATE_IP_0> slots=1
<PRIVATE_IP_1> slots=1
EOF
mpirun -np 2 -hostfile hosts ./mc_mpi_cpu --scenarios 50000

## Destroy
terraform destroy -var-file=terraform.tfvars -auto-approve
