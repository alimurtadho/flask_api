import numpy
import pandas
import matplotlib.pyplot as plot

def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=

    Example: ["sex == 1", 'age < 18']
    """

    field, op, value = condition.split(" ")

    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")

    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")

    # filter data and outcomes
    data = data[matches].reset_index(drop = True)
    return data

def disease_stats(data, outcomes, key, filters = []):
    """
    Print out selected statistics regarding survival, given a feature of
    interest and any number of filters (including no filters)
    """

    # Check that the key exists
    if key not in data.columns.values :
        print("'{}' is not a feature of the data.".format(key))
        return False

    # Merge data and outcomes into single dataframe
    all_data = pandas.concat([data, outcomes], axis = 1)

    # Apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # Create outcomes DataFrame
    all_data = all_data[[key, 'num']]

    # Create plotting figure
    plot.figure(figsize=(8,6))

    # 'Numerical' features
    if(key == 'age' or key == 'trestbps' or key == 'chol' or key == 'thalach' or key == 'oldpeak'):
	
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value
		
        bins = numpy.arange(0, all_data[key].max() + 10, 10)
		
        diseased_vals = all_data[all_data['num'] != 0][key].reset_index(drop = True)
        non_diseased_vals = all_data[all_data['num'] == 0][key].reset_index(drop = True)
        plot.hist(diseased_vals, bins = bins, alpha = 0.6,
                 color = 'red', label = 'Having Disease')
        plot.hist(non_diseased_vals, bins = bins, alpha = 0.6,
                 color = 'green', label = 'Not Having Disease')

        # Add legend to plot
        plot.xlim(0, bins.max())
        plot.legend(framealpha = 0.8)

    # 'Categorical' features
    else:

        # Set the various categories
        if(key == 'sex'):
            values = numpy.arange(2)
        if(key == 'cp'):
            values = numpy.arange(1,5)
        if(key == 'fbs'):
            values = numpy.arange(2)
        if(key == 'restecg'):
            values = numpy.arange(3)
        if(key == 'exang'):
            values = numpy.arange(2)
        if(key == 'slope'):
            values = numpy.arange(1,4)
        if(key == 'ca'):
            values = numpy.arange(4)
        if(key == 'thal'):
            values = [3,6,7]

        # Create DataFrame containing categories and count of each
        frame = pandas.DataFrame(index = numpy.arange(len(values)), columns=(key,'NDiseased','Diseased'))
        for i, value in enumerate(values):
            frame.loc[i] = [value, \
                   len(all_data[(all_data['num'] == 0) & (all_data[key] == value)]), \
                   len(all_data[(all_data['num'] != 0) & (all_data[key] == value)])]

        # Set the width of each bar
        bar_width = 0.4

        # Display each category's survival rates
        for i in numpy.arange(len(frame)):
            diseased_bar = plot.bar(i-bar_width, frame.loc[i]['Diseased'], width = bar_width, color = 'r')
            non_diseased_bar = plot.bar(i, frame.loc[i]['NDiseased'], width = bar_width, color = 'g')

            plot.xticks(numpy.arange(len(frame)), values)
            plot.legend((diseased_bar[0], non_diseased_bar[0]),('Having Disease', 'Not Having Disease'), framealpha = 0.8)

    # Common attributes for plot formatting
    plot.xlabel(key)
    plot.ylabel('Number of Patients')
    plot.title('Patients Disease Statistics With \'%s\' Feature'%(key))
    plot.show()