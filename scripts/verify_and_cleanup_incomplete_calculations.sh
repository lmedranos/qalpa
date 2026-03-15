#!/bin/bash

################################################################################
# Verification and Cleanup Script for Incomplete Calculations
# 
# This script:
# 1. Scans directory structure for incomplete aims.out files
# 2. Identifies corresponding .xyz files
# 3. Optionally deletes incomplete .xyz files based on DELETE_FLAG
#
# Usage: bash verify_and_cleanup_incomplete_calculations.sh
# or as SLURM job: sbatch verify_and_cleanup_incomplete_calculations.sh
################################################################################

#SBATCH --job-name=verify-cleanup-calcs
#SBATCH --output=verify-cleanup-%j.log
#SBATCH --error=verify-cleanup-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

################################################################################
# USER CONFIGURATION - MODIFY THESE
################################################################################

# Main directory containing target folders
# e.g., /path/to/data_vanilla_full_dataset/evaluation
# or   /path/to/data_full_dataset/evaluation
MAIN_DIR="/data/horse/ws/miha493e-research_project/activediffusion/data/data_vanilla_full_dataset/evaluation"

# Flag to control deletion: true=delete, false=report only
DELETE_FLAG=false

# Model range to scan (0-based index): e.g., "0-2" or "0-8" or just "0"
MODEL_RANGE="0-2"

# Output directory for reports
OUTPUT_DIR="${MAIN_DIR}/verification_reports"

################################################################################
# END USER CONFIGURATION
################################################################################

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Timestamp for file naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/verification_report_${TIMESTAMP}.txt"
INCOMPLETE_FILE="${OUTPUT_DIR}/incomplete_files_${TIMESTAMP}.txt"
DELETION_LOG="${OUTPUT_DIR}/deletion_log_${TIMESTAMP}.txt"

# Initialize counters
TOTAL_XYZ=0
COMPLETE_XYZ=0
INCOMPLETE_XYZ=0

# Color codes for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

################################################################################
# FUNCTIONS
################################################################################

log_message() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg"
}

print_header() {
    local title="$1"
    echo "================================================================================"
    echo "$title"
    echo "================================================================================"
}

check_aims_complete() {
    local aims_file="$1"
    
    # Check if file exists
    if [[ ! -f "$aims_file" ]]; then
        echo "NOT_FOUND"
        return 1
    fi
    
    # Check if file ends with "Have a nice day"
    if tail -c 500 "$aims_file" 2>/dev/null | grep -q "Have a nice day"; then
        echo "COMPLETE"
        return 0
    else
        echo "INCOMPLETE"
        return 1
    fi
}

################################################################################
# MAIN SCRIPT
################################################################################

{
    print_header "Incomplete Calculation Verification and Cleanup"
    log_message "Start time: $(date)"
    log_message "Main directory: $MAIN_DIR"
    log_message "Delete flag: $DELETE_FLAG"
    log_message "Model range: $MODEL_RANGE"
    echo ""
    
    # Validate main directory
    if [[ ! -d "$MAIN_DIR" ]]; then
        echo -e "${RED}ERROR: Main directory does not exist: $MAIN_DIR${NC}"
        log_message "ERROR: Main directory does not exist" >> "$REPORT_FILE"
        exit 1
    fi
    
    log_message "Directory validation: OK"
    
    # Parse model range
    IFS='-' read -r MODEL_START MODEL_END <<< "$MODEL_RANGE"
    if [[ -z "$MODEL_END" ]]; then
        MODEL_END="$MODEL_START"
    fi
    
    log_message "Scanning models $MODEL_START to $MODEL_END..."
    echo ""
    
    # Arrays to track files
    declare -a INCOMPLETE_COORDS
    declare -a INCOMPLETE_XYZ_FILES
    
    # Iterate through targets and models
    for target_dir in "${MAIN_DIR}"/target*/; do
        if [[ ! -d "$target_dir" ]]; then
            continue
        fi
        target_name=$(basename "$target_dir")
        
        for model_num in $(seq "$MODEL_START" "$MODEL_END"); do
            model_dir="${target_dir}/model${model_num}"
            
            if [[ ! -d "$model_dir" ]]; then
                continue
            fi
            
            query_dir="${model_dir}/query"
            prop_calc_dir="${model_dir}/prop_calc"
            
            if [[ ! -d "$query_dir" ]] || [[ ! -d "$prop_calc_dir" ]]; then
                continue
            fi
            
            # Find all .xyz files in query directory
            while IFS= read -r -d '' xyz_file; do
                ((TOTAL_XYZ++))
                
                # Extract basename without extension
                xyz_basename=$(basename "$xyz_file" .xyz)
                
                # Path to corresponding aims.out
                aims_out_file="${prop_calc_dir}/${xyz_basename}/aims.out"
                
                # Check if aims.out is complete
                completeness=$(check_aims_complete "$aims_out_file")
                
                if [[ "$completeness" == "COMPLETE" ]]; then
                    ((COMPLETE_XYZ++))
                else
                    ((INCOMPLETE_XYZ++))
                    relative_path="${target_name}/model${model_num}/query/${xyz_basename}.xyz"
                    INCOMPLETE_XYZ_FILES+=("$xyz_file")
                    INCOMPLETE_COORDS+=("$relative_path")
                    echo "$relative_path (status: $completeness)" >> "$INCOMPLETE_FILE"
                fi
            done < <(find "$query_dir" -maxdepth 1 -name "*.xyz" -print0)
        done
    done
    
    # Print summary
    print_header "Verification Summary"
    echo "Total .xyz files scanned:    $TOTAL_XYZ"
    echo "Complete calculations:        $COMPLETE_XYZ ($(awk "BEGIN {printf \"%.1f\", $COMPLETE_XYZ*100/$TOTAL_XYZ}")%)"
    echo "Incomplete calculations:      $INCOMPLETE_XYZ ($(awk "BEGIN {printf \"%.1f\", $INCOMPLETE_XYZ*100/$TOTAL_XYZ}")%)"
    echo ""
    
    if [[ $INCOMPLETE_XYZ -eq 0 ]]; then
        echo -e "${GREEN}✓ All calculations are complete!${NC}"
    else
        echo -e "${YELLOW}⚠ Found $INCOMPLETE_XYZ incomplete calculations${NC}"
        echo ""
        print_header "Incomplete Files"
        cat "$INCOMPLETE_FILE"
        echo ""
    fi
    
    # Handle deletion if flag is set and incomplete files exist
    if [[ "$DELETE_FLAG" == "true" ]] && [[ $INCOMPLETE_XYZ -gt 0 ]]; then
        print_header "Deletion Process"
        log_message "DELETE_FLAG is true. Proceeding with deletion..."
        
        {
            echo "Deletion Log - $(date)"
            echo "========================================"
            echo "Total files to delete: $INCOMPLETE_XYZ"
            echo ""
        } > "$DELETION_LOG"
        
        deleted_count=0
        failed_count=0
        
        for xyz_file in "${INCOMPLETE_XYZ_FILES[@]}"; do
            if [[ -f "$xyz_file" ]]; then
                rm -f "$xyz_file"
                if [[ $? -eq 0 ]]; then
                    ((deleted_count++))
                    echo "✓ Deleted: $xyz_file" | tee -a "$DELETION_LOG"
                else
                    ((failed_count++))
                    echo "✗ Failed to delete: $xyz_file" | tee -a "$DELETION_LOG"
                fi
            else
                ((failed_count++))
                echo "✗ File not found: $xyz_file" | tee -a "$DELETION_LOG"
            fi
        done
        
        echo "" | tee -a "$DELETION_LOG"
        print_header "Deletion Summary"
        echo "Successfully deleted:  $deleted_count"
        echo "Failed to delete:      $failed_count"
        echo -e "New completion rate:   $((COMPLETE_XYZ + deleted_count))/$TOTAL_XYZ ($(awk "BEGIN {printf \"%.1f\", ($COMPLETE_XYZ + $deleted_count)*100/$TOTAL_XYZ}")%)"
        
        {
            echo ""
            echo "Successfully deleted: $deleted_count"
            echo "Failed to delete:     $failed_count"
            echo "New total:            $((COMPLETE_XYZ + deleted_count))/$TOTAL_XYZ"
        } >> "$DELETION_LOG"
    elif [[ "$DELETE_FLAG" == "false" ]] && [[ $INCOMPLETE_XYZ -gt 0 ]]; then
        echo -e "${YELLOW}Deletion is disabled (DELETE_FLAG=false)${NC}"
        echo "To enable automatic deletion, set DELETE_FLAG=true at the top of this script"
    fi
    
    # Final summary
    print_header "Completion Summary"
    log_message "End time: $(date)"
    log_message "Verification complete"
    echo ""
    echo "Output files generated:"
    if [[ -f "$REPORT_FILE" ]]; then
        echo "  - Report: $REPORT_FILE"
    fi
    if [[ -f "$INCOMPLETE_FILE" ]] && [[ $INCOMPLETE_XYZ -gt 0 ]]; then
        echo "  - Incomplete files list: $INCOMPLETE_FILE"
    fi
    if [[ -f "$DELETION_LOG" ]]; then
        echo "  - Deletion log: $DELETION_LOG"
    fi
    
} | tee "$REPORT_FILE"

log_message "Script execution completed"
exit 0
