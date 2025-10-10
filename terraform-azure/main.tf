resource "random_id" "suffix" { byte_length = 3 }

resource "azurerm_resource_group" "rg" {
  name     = "${var.resource_group_name_prefix}-${random_id.suffix.hex}"
  location = var.location
}

resource "azurerm_virtual_network" "vnet" {
  name                = "mc-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_subnet" "subnet" {
  name                 = "mc-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_network_security_group" "nsg" {
  name                = "mc-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "SSH"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "MPI"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["8000-9000"]
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "pip" {
  count               = var.vm_count
  name                = "mc-pip-${count.index}"
  allocation_method   = "Static"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Basic"
}

resource "azurerm_network_interface" "nic" {
  count               = var.vm_count
  name                = "mc-nic-${count.index}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "ipconfig"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pip[count.index].id
  }
}

resource "azurerm_network_interface_security_group_association" "assoc" {
  count                     = var.vm_count
  network_interface_id      = azurerm_network_interface.nic[count.index].id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

data "local_file" "ssh_key" { filename = var.ssh_public_key_path }

locals {
  startup = <<-CLOUD
    #cloud-config
    package_update: true
    packages:
      - build-essential
      - git
      - mpich
    runcmd:
      - su - ${var.admin_username} -c "git clone ${var.repo_url} ~/hpc-montecarlo || true"
      - su - ${var.admin_username} -c "cd ~/hpc-montecarlo/src || cd ~/hpc-montecarlo; if [ -f src/mc_mpi_cpu.cpp ]; then cd src; mpicxx -O3 -std=c++17 mc_mpi_cpu.cpp -o mc_mpi_cpu || true; fi"
  CLOUD
}

resource "azurerm_linux_virtual_machine" "vm" {
  count                 = var.vm_count
  name                  = "mc-vm-${count.index}"
  location              = azurerm_resource_group.rg.location
  resource_group_name   = azurerm_resource_group.rg.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [ azurerm_network_interface.nic[count.index].id ]

  admin_ssh_key {
    username   = var.admin_username
    public_key = data.local_file.ssh_key.content
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    name                 = "mc-osdisk-${count.index}"
  }

  custom_data = base64encode(local.startup)
}
