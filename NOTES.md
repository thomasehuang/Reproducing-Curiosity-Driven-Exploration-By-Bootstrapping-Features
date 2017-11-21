Notes from running script
=========================
- Use 10vCPUs with 20GB of memory
- Best_cost seems to not be assigned
- Saving takes a long time, instead of rewriting an array just append to a file
- Error at end of cbf.py's execution:
	```
	Traceback (most recent call last):
	  File "cbf.py", line 179, in <module>
	  File "cbf.py", line 150, in cbf
	    for graph_epi_len, timestep in graph_epi_lens:
	TypeError: not all arguments converted during string formatting
	```
