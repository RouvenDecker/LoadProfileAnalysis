clean:
	rm -rf data
	mkdir data
	rm -rf output
	mkdir output

run:
	./LoadProfileAnalysis.py

run_csv:
	./LoadProfileAnalysis.py --csv True
