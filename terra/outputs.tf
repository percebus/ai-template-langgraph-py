
output "environment" {
  value = local.environment
}

output "stack" {
  value = {
    id  = random_id.stack.id
    hex = random_id.stack.hex
  }
}

output "ARM_region" {
  value = local.location
}

output "ARM_resource_group" {
  value = {
    id   = azurerm_resource_group.rg.id
    name = azurerm_resource_group.rg.name
  }
}
