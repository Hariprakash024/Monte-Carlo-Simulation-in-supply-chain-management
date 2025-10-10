output "resource_group" { value = azurerm_resource_group.rg.name }
output "public_ips"     { value = [for p in azurerm_public_ip.pip : p.ip_address] }
output "private_ips"    { value = [for n in azurerm_network_interface.nic : n.ip_configuration[0].private_ip_address] }
output "ssh_commands"   { value = [for p in azurerm_public_ip.pip : "ssh ${var.admin_username}@${p.ip_address}"] }
