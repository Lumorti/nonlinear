#!/bin/bash

# Start with d=2 and go up to d=5
initd=2
finald=5

# Number of repeats per
repeats=3

# Clear the output file
> scaling.dat

# For each dimension
for d in $(seq $initd $finald)
do

	# For each repeat
	total=0;
	for repeat in $(seq 1 $repeats)
	do

		# Run the system
		val=$(./nl -d $d -n 2 -K -R)

		# Ouptut to terminal
		echo "repeat $repeat for d=$d in $val ms"

		# Add to the total for this dimension
		total=$(echo "$total+$val" | bc)

	done

	# Calculate the average
	avg=$(echo "$total / $repeats" | bc)

	# Ouptut to terminal
	echo "average was $avg ms"

	# Write to file
	echo "$d $avg" >> scaling.dat

done

# Plot the output
