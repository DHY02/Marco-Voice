
#!/bin/bash

cd batch_inference_results

for file in uttr_*_*.wav; do
    if [[ -f "$file" ]]; then
        # Use sed to remove the uttr_XXX_ pattern
        new_name=$(echo "$file" | sed 's/^uttr_[0-9]*_//')
        
        if [[ "$new_name" != "$file" ]]; then
            echo "Renaming: $file -> $new_name"
            mv "$file" "$new_name"
        fi
    fi
done

echo "Done!"