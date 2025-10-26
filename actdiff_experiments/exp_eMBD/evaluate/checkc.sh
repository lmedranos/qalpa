#!/bin/bash
evaluation="/data/horse/ws/miha493e-research_project/activediffusion/data/data_embd_non_opt2/evaluation"

output_file="check.txt"
echo "Results summary" > "$output_file"
echo "=========================" >> "$output_file"

# Loop through all target/model combinations
for target in {0..3}; do
  for model in {0..8}; do
    dir=$evaluation"/target${target}/model${model}/prop_calc"

    if [ -d "$dir" ]; then
      echo "Directory: $dir" >> "$output_file"

      # Count completed runs
      completed=$(find "$dir" -type f -name "aims.out" \
        -exec sh -c 'tail -n 20 "$1" | grep -q "Have a nice day." && dirname "$1"' _ {} \; \
        | sort -u | wc -l)

      # Count incomplete runs
      incomplete=$(find "$dir" -type f -name "aims.out" \
        -exec sh -c '! tail -n 20 "$1" | grep -q "Have a nice day." && dirname "$1"' _ {} \; \
        | sort -u | wc -l)

      echo "  Completed:   $completed" >> "$output_file"
      echo "  Incomplete:  $incomplete" >> "$output_file"

      # List incomplete files and check for corresponding .xyz files
      if [ "$incomplete" -gt 0 ]; then
        echo "  Incomplete files:" >> "$output_file"
        find "$dir" -type f -name "aims.out" \
          -exec sh -c '! tail -n 20 "$1" | grep -q "Have a nice day." && dirname "$1"' _ {} \; \
          | sort -u | while read -r incdir; do
              echo "    - $incdir" >> "$output_file"

              # Extract run name (e.g., run_12)
              run_name=$(basename "$incdir")

              # Build the corresponding .xyz file path
              xyz_file=$evaluation"/target${target}/model${model}/query/${run_name}.xyz"

              # Check if .xyz file exists
              if [ -f "$xyz_file" ]; then
                echo "        → Found: $xyz_file" >> "$output_file"
              else
                echo "        → Missing: $xyz_file" >> "$output_file"
              fi
            done
      fi

      echo "" >> "$output_file"
    else
      echo "Directory: $dir (not found)" >> "$output_file"
      echo "" >> "$output_file"
    fi
  done
done

echo "Done! Results saved in $output_file."
