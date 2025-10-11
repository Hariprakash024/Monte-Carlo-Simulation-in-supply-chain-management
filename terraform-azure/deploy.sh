#!/bin/bash

# Define an array of ALL valid Azure region slugs from the error message.
# Note: The order doesn't matter much here since we are just checking for the first allowed one.
ALLOWED_REGIONS=(
    "australiacentral" "australiacentral2" "australiaeast" "australiasoutheast"
    "austriaeast" "belgiumcentral" "brazilsouth" "brazilsoutheast" "brazilus"
    "canadacentral" "canadaeast" "centralindia" "centralus" "centraluseuap"
    "chilecentral" "denmarkeast" "eastasia" "eastus" "eastus2" "eastus2euap"
    "eastus3" "eastusslv" "francecentral" "francesouth" "germanynorth"
    "germanywestcentral" "indiasouthcentral" "indonesiacentral" "israelcentral"
    "israelnorthwest" "italynorth" "japaneast" "japanwest" "jioindiacentral"
    "jioindiawest" "koreacentral" "koreasouth" "malaysiasouth" "malaysiawest"
    "mexicocentral" "newzealandnorth" "northcentralus" "northeurope"
    "norwayeast" "norwaywest" "northeastus5" "polandcentral" "qatarcentral"
    "southafricanorth" "southafricawest" "southcentralus" "southcentralus2"
    "southeastasia" "southeastus" "southeastus3" "southeastus5" "southindia"
    "southwestus" "spaincentral" "swedencentral" "swedensouth" "switzerlandnorth"
    "switzerlandwest" "uaecentral" "uaenorth" "uksouth" "ukwest"
    "westcentralus" "westeurope" "westindia" "westus" "westus2" "westus3"
)

# Define the path to your variables file
VARS_FILE="terraform.tfvars"

echo "Starting automated Terraform deployment with full regional fallback (${#ALLOWED_REGIONS[@]} regions)..."
echo "--------------------------------------------------------"

# Loop through the list of regions
for REGION in "${ALLOWED_REGIONS[@]}"; do
    echo "Attempting deployment in region: $REGION"
    
    # 1. Update the location variable in terraform.tfvars
    # This command uses sed to replace the existing location line with the new region.
    sed -i "s/^location = \".*\"/location = \"$REGION\"/" "$VARS_FILE"

    echo "Updated $VARS_FILE with location = \"$REGION\""

    # 2. Run terraform apply
    # We pipe the output to a temporary file to check for failure messages easily
    terraform apply -var-file="$VARS_FILE" -auto-approve 2>&1 | tee /tmp/tf_apply_output.log
    APPLY_STATUS=${PIPESTATUS[0]}

    # Check the exit status of the apply command (0 means success)
    if [ $APPLY_STATUS -eq 0 ]; then
        echo "--------------------------------------------------------"
        echo "✅ SUCCESS! Infrastructure deployed in $REGION."
        echo "--------------------------------------------------------"
        # Exit the script with success code
        exit 0
    else
        # Check for the specific policy error (403 Forbidden)
        if grep -q "403 Forbidden" /tmp/tf_apply_output.log; then
            echo "❌ Policy Restriction: $REGION failed due to 403 Forbidden (Policy). Retrying with next region."
            echo "--------------------------------------------------------"
        else
            echo "❌ CRITICAL FAILURE: $REGION failed for a reason other than policy (e.g., syntax error or authentication issue)."
            echo "Check /tmp/tf_apply_output.log for details."
            echo "--------------------------------------------------------"
            # If it failed for a non-policy reason, stop the script
            exit 1
        fi
    fi
done

echo "--------------------------------------------------------"
echo "🛑 FAILURE: Exhausted all fallback regions. None are allowed by policy."
echo "--------------------------------------------------------"

# Exit the script with failure code
exit 1
