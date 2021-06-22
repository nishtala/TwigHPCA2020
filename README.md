## I'm no longer maintaing this project. Please fix any bugs you find yourself 🦖
* I. Overview

Twig is a user-level scheduler for Linux OS, designed to implement a deep RL-based BDQ architecture for scheduling latency-critical applications and improving energy efficiency of cloud systems. This scheduler was built at the Norwegian University of Science and Technology, and is provided with a copy of the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

Very important! Please read (at the very least browse through) the following [paper](https://sites.google.com/view/rnishtala/publications) before using Twig.
The paper will explain the purpose of our scheduler and how it works.

* II. How to install

1. Here is how to install packages necessary for Twig.

 - sudo pip3 install -r requirements.txt
 - Please install the latest version of [libpfm](http://perfmon2.sourceforge.net/), and its Python API.

2. How to get initial input parameters
 
 To give the basic input details for Twig. Please identify the following and provide it as an input in `common.py`; 

 A placeholder (`None`) for these variables is provided as part of the file.
 - Normalisation:      Run the three microbenchmarks consecutively (provided) on all cores at the highest DVFS state and gather the average of each of the 11 counters and power consumption, as specified in Section 4.
 - LC workload :       Name, PPID, Max load, and target of the latency-critical workloads
 - Sampling frequency: Provide the sampling frequency
 - Power model coefs : As specified in Section IV, we use the power model to determine the reward. Please compute the parameters and provide it as a input.
 - Reward coefs :      For best use of Twig, we expect the user to do a study on the co-efficients and provide reasonably accurate values

* III. How to run

 As given in `twig_control.py`, provide:
 - the reported power consumption, the 99th %-ile latency, and recorded power consumption, at each sampling interval

In case you will have additional questions (especially about internal code of the scheduler), please don't hesitate to email me directly at: <rajiv.nishtala@ntnu.no> or <nishtala.raj@gmail.com>

* IV. Getting help

Got a question? Found a bug? Please contact me directly.  

Please do the following to get a useful response and save time for both of us:

1) Please briefly describe what do you want to use Twig for? What is the purpose of your experiments? Without understanding what you want to see, it is hard to recommend the best use of Twig for your task. Also, please elaborate a little bit on the workload you are using in your tests (what apps, what is their CPU usage, etc.). 

2) Please indicate what Twig version you're using, and send the output file from Twig.

* V. Citation

If you do use Twig, we request you to cite the Publication

    @INPROCEEDINGS{Nishtala_HPCA2020,
    author={R. {Nishtala} and V. {Petrucci} and P. {Carpenter} and  M. {Sjalander}},
    booktitle={2020 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
    title={{Twig: Multi-agent Task Management for Colocated latency-critical Cloud Services}},
    year={2020},
    }

Cheers!
- Rajiv
