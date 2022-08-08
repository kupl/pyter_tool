# Contents of our PyTER

* Tools `(~/my_tool)`
* Test information `(~/test_info/*.json)` : Specifying which is negative cases and positive cases.  
* Pyannotate (https://github.com/dropbox/pyannotate) `(~/pyannotate)`

# How to run arbitrary test cases

I highly recommend to make a setting via [pyenv](https://github.com/pyenv/pyenv) and [virtualenv](https://github.com/pyenv/pyenv-virtualenv) instead of local environment.

I assuem the situation that I try to run arbitrary project.
This project has testfile named `negative.py` and its content is following form:

```
# negative.py

class Negative() :
  ...
  
  def negative_test1() :
    ...
    
  def negative_test2() :
    ...
```

You first set your project like this:

```
<benchmark name>
L <project>-<id>
  L ...
  L <test folder name>
    L <negative.py>
```

### Make a test information

You first make a test information file such as `(~/test_info/*.json)`:

```
{
    ...,
    
    "<project>-<id>" : {
        "all" : [
            "--continue-on-collection-errors", 
            "--execution-timeout", "300", 
            "--ignore=<test folder name>/<negative.py>",
            "<test folder name>"
        ],
        "pos" : [
            "--continue-on-collection-errors", 
            "--execution-timeout", "300", 
            "-k", "not (<negative_test1> or <negative_test2>)",
            "<test folder name>/<negative.py>"
        ],
        "neg" : [
            "--continue-on-collection-errors", 
            "--execution-timeout", "300", 
            "<test folder name>/<negative.py>::<Negative>::<negative_test1>",
            "<test folder name>/<negative.py>::<Negative>::<negative_test2>",
            ...
        ]
    },
    
    ...
}
```

Let assume folders consists of like this:

```
<benchmark name>
L <project>-<id>
  L ...
  L <test folder name>
    L <negative.py>
<test info name>
L <project>.json
```

### Run the dynamic analysis

First, let move to the project folder:

```
cd <project path>
```
and install several pytest libraries and PyAnnotate for the dynamic analysis:

```
pip install pytest
pip install pytest-timeouts
pip install -e /<pyter path>/pyannotate/.
```

Then, you can run the dynamic analysis as following code:

```
python /<pyter path>/my_tool/extract_neg.py --bench=<test info name>
python /<pyter path>/my_tool/extract_pos.py --bench=<test info name>
```

The result of the dynamic analysis will be saved in `/<benchmark name>/<project>-<id>/pyter` folder.

### Generate patches from PyTER

You can run our PyTER framework as following code:

```
python /<pyter path>/my_tool/test_main.py -d "<benchmark name>" -p "<project>" -i "<id>" -c ""
```

