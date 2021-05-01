"""Useful functions for the Project 1."""

SAMPLE_FRAC = 0.13     # A fraction of dataset to subsample during the development (set to 1.00 for the actual final computation)

def find_encoding(fpath):
    """
    Determine and print out the encoding.
    INPUT - fpath - path to a csv file with unknown encoding

    from: https://stackoverflow.com/questions/436220/how-to-determine-the-encoding-of-text

    """
    import chardet     # import the chardet library

    # use the detect method to find the encoding
    # 'rb' means read in the file as binary
    with open(fpath, 'rb') as file:
        print(chardet.detect(file.read()))

def rename_top_level(df):
    """
    Renames columns that are missing from the top level name to fix the multiindex.
    INPUT - A dataframe loaded from a CSV file.
    OUTPUT - A dataframe with fixed names of the columns arranged in multiindex.
    """
    mapping = {}
    for i, col in enumerate(df.columns):
        if 'Unnamed:' in col[0]:
            mapping[col[0]] = previous    # Use the name of the previous column
        else: previous = col[0]

    df = df.rename(columns=mapping, level=0)

    return df

def merge_multichoice(df):
    """
       Several columns in the datasets encode different answers to the same multiple-choise questions (using dummy variables).
       This function merges all these answers to a list separated with semicolons and stores them in a new single column.
       Answers to open-ended questions are ommited.
       INPUT - a dataframe with several columns encodinganswers to multiple choice questions. The column names must be arranged
               as a multiindex with matching top level for each separate question.
       OUTPUT - a dataframe with merged answers and collapsed column index
    """

    # Find all columns that store only single unique value
    single_answer_cols = [col for col in df.columns if len(df[col].value_counts().index) == 1]
    # Collapse multiple-choice questions and drop the original columns
    for col_lvl0 in set([col[0] for col in single_answer_cols]):
        cols = [col for col in single_answer_cols if col[0] == col_lvl0]    # All columns that have the same top level in the multindex (and contain a single unique value in the responses)
        if len(cols) > 1:
            df[col_lvl0 + ' (MERGED)'] = df[cols].apply(lambda row : '; '.join(row.dropna().tolist()), axis=1).replace('', np.nan)
            df = df.drop(columns=cols)

    # Flatten the multiindex in columns
    df.columns = [' - '.join(col).strip() if (col[1] != 'Response' and len(col[1])>0) else col[0].strip() for col in df.columns]

    return df

def replace_choices(row, mapping):
    """
        Replaces (entire, not just substrings) multiple choices with their equivalents (substitutes) from a dictionary.

        INPUTS: row - a string of choices separated by semicolons
                mapping - a dictionary of substrings to be replaced (in the form old(key)<->new(val))

        OUTPUT: cleaned list with entries replaced according to the mapping
    """
    if pd.isnull(row):
        return row

    # Convert the string into a list of separate choices
    row = list(map(str.strip, row.split(';')))

    # Replace each entry in the list of choises by its substitute from the mapping dictionary
    for i, s in enumerate(row):
        try:
            row[i] = mapping[s]
        except KeyError: pass

    # Make sure that each choice is included only once and separated by '; ' without extra whitespace. Remove empty entries
    result = '; '.join( filter(None, set(row)) )

    # Make sure that empty strings are interpreted as nulls
    if len(result) == 0: result = None

    return result

def list_choices(ser):
    """
    Returns all possible answers to single- or multiple-choice questions.
    INPUTS:
        ser - a series with answers to multiple choice questions sepoarated by ';'
    OUTPUT:
        a list of strings corresponding to all different choices.
    """
#     Join all entries in the series into a single long string, and then split it and convert to a set
    return sorted(list( set(map(str.strip, set(';'.join(ser.dropna().values).split(';') ) )) ))

def join_entries(df, columns, sep=';'):
    """
       Joins entries in several columns in a dataframe into a single string.
       INPUTS:
           df - the source dataframe
           columns - a list of coulmns to join
           sep - separator to use between substrings taken from different columns
        OUTPUT:
            A series with joined substrings.
    """
    return df[columns].apply(lambda row : sep.join(row.dropna().tolist()), axis=1).replace('', np.nan)

def expand_multichoice(df, column=None, drop=True, rename=False):
    """
    Expands a column containing answers to a multiple choice question into
    several boolean columns corresponding to each choice (resulting in a one-hot encoding).
    Similar to pd.get_dummies, but also works for multiple-choice questions.

    INPUTS:
        df - a dataframe to work with
        column - a column name containing multiple choice answers separated by ';';
                also works for single-choice questions as well.
        drop - if True removes the original column from the dataframe
        rename - if True, the names of the created columns will be set to tuples
                with the first element equal to the name of the original column, and
                the second - each of the multiple choices
    OUTPUT:
        The updated dataframe with expanded columns.
    """
    df = df.copy()     # Copy the dataframe to prevent it from being modified in-place

    # If it's a Series, turn it into a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
        column = df.columns[0]
    elif column is None:
        raise RuntimeError('Column name must be specified.')

    all_choices = list_choices(df[column])      # All possible choices
#     is_multichoice = (df[column].str.contains(';', regex=False)).any()  # true if the column represents a multiple-choice question

    df[column] = df[column].str.replace(r'\s*;\s*', ';')   # Remove all whitespaces around semicolons, if any
    col_split = df[column].str.split(';')                  # Store the split strings to save time

    for ch in all_choices:
        # Choose the names for the new columns
        if (not drop) or rename:
            name = (column, ch)
        else:
            name = ch

        if sum(map(lambda x : (ch in x), all_choices)) == 1:
            # If it is a single-choice column or if the current pattern ch occurs only once in the list of all possible choices use a simple str function -- it is faster
            df[name] = df[column].str.contains(ch, regex=False)
        else:
            # if the pattern occurs as a substring in more than one possible choices (e.g. 'C' in 'C++' and 'C#'), split the string first and process each entry individually -- this takes longer
            df[name] = col_split.map(lambda x : (ch in x), na_action='ignore') # .apply(lambda x : ch in list(map(str.strip, x)) )

    if drop:
        df.drop(columns=[column], inplace=True)

    return df

def counts_by_group(df, target_col, group_col='SurveyYear', total=True, sort_by_total=False):
    """
        Counts the number of occurences of a different entries in a column by year.
        Works with columns corresponding to single- and multiple-choice questions.
        INPUTS:
                df - a dataframe to get the data from
                target_col - which column to count
                group_col - a column to group the entries by.
                total - if True, adds a row with total number of non-NA entries in each group
                sort_by_total - if True, will sort the rows in the final dataframe by the total number of occurences across all years.
        OUTPUTS:
                df - the dataframe of counts; the rows correspond to individual entries in the target column,
                     and the columns are unstacked entries of the group_by column (e.g. years).
    """

    dfExp = expand_multichoice(df[[group_col, target_col]], column=target_col)   # An expanded dataframe with new columns corresponding to different choices
    dfCnt = dfExp.groupby(group_col).sum(axis=0).T        # dataframe with the counts arranged by group
    dfCnt.index.rename(target_col, inplace=True)       # Rename the index

    # Add the row with toal number of non-NA entries in each group
    if total:
        dfCnt = dfCnt.append( df[[group_col, target_col]].groupby(group_col).count().squeeze().rename('TOTAL') )     # Number of non-NA values in each group (year)

    return dfCnt

def bars_by_year(df, column, ntop=10, sortkey=None, color=None, title=None, ax=None):
    """
        Plots a histogram of the top occurences in a certain column by year.
        INPUTS:
                df - a dataframe to get the data from
                column - which column to plot
                ntop - how many top entries to consider
                color - a sequence of colors, as in pandas.plot.bar
                ax - a handle to axis to plot on
                sortkey - a callable or a mapping by which to sort the categories, takes values from the index and returns an ordinal (e.g. an integer)
        OUTPUTS:
                ax - the handle to the plotted figure;
                by_year - the full dataframe of the results.
    """
    # create a dataframe of different occurences by year
    by_year = df.groupby('SurveyYear')[column].value_counts().unstack('SurveyYear')
    by_year['Total'] = by_year.sum(axis=1)
    by_year /= by_year.sum()    # Turn into ratios
    by_year.sort_values('Total', ascending=False, inplace=True)

    # Additionally, sort according to the sortkey, if provided
    if sortkey is not None:
        by_year['_sortKey'] = by_year.index.map(sortkey)     # An auxillary column to sort by (can yse the 'key' option in Pandas > 1.1.0)
        by_year.sort_values('_sortKey', inplace=True)
        by_year.drop('_sortKey', axis=1, inplace=True)

    # Reset the colors
    if color is None:
#         color = cm.get_cmap('Set1')(np.arange(min(9, ntop))/9)
        color = cm.ScalarMappable(cmap='jet').to_rgba(np.linspace(0, 1, min(ntop,by_year.shape[0]) ))

    ax = by_year.head(ntop).T.plot.bar(stacked=True, ax=ax, color=color)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title if title is not None else column)

    return ax, by_year

def median_from_ranges(ranges, counts=None):
    """
    Computes an (approximate) median value given a list of ranges (e.g. age brackets).
    Main idea: Represent each range as a cummulative function, combine them together
    and find the value corresponding to 0.5 of this function.
    Assuming that the distribution in each range is uniform, this amounts to combining several linear ramp functions.

    INPUTS:
        ranges - list of 2-tuples, each of which defines min and max values for the ranges;
        counts - list of integers specifying how often each range occurs in the dataset; if None, set to all ones.

    OUTPUT:
        median - the estimated value of the median computed using an approximate entire distribution of the variable by combining its ranges.
    """
    if counts is None:
        counts = [1]*len(ranges)
    elif len(ranges) != len(counts):
        raise RuntimeError('The number of counts must be equal the number of ranges.')

    # Compute the node-points of the future cummulative distribution (they correspond to the limits of the ranges)
    nodes = sorted(list(set( chain(*ranges) )))

    cumfun = np.array([np.interp(nodes, xp=rng, fp=[0, cnt] ) for rng, cnt in zip(ranges, counts) ]).sum(axis=0)     # Cummulative functions for each a range evaluated at all nodes and then summed together to result in the total cummulative function
    cumfun /= sum(counts)     # Normalize to 1

    # Use interpolation to find the 0.5 value of the cummulative distribution (treat cumfun values as x and nodes as y)
    result = np.interp(0.5, xp=cumfun, fp=nodes)

    return result

def parse_bracket(r_str, d=0.5):
    """
    Converts a range bracket in the form "x to y" to a tuple of min and max values.
    If integer is passed, returns a range of +/-d around that number (by default, d=0.5).

    INPUT: r_str - a string in the form "x to y"

    OUTPUT: r_tup - a tuple (x, y)
    """
    try:
        x = int(r_str)
        r_tup = (x-d, x+d)
    except ValueError:
        if 'to' in r_str:
            r_tup = tuple([int(x) for x in re.split(r'\sto\s', r_str)])
    return r_tup

def agg_median(ser):
    """
    A function to compue the approximate median from ranges,
    which can be used on a data frame with pd.DataFrame().aggregate.

    INPUTS: ser - a series, whose index contains different bracket/range
                  definitions and values are the number of their occurences(e.g. output by value_counts)
    OUTPUT: an approximate median as returned by the median_from_ranges function
    """
    ser = ser.dropna()
    result = median_from_ranges(ranges=ser.index.map(parse_bracket),
                              counts=ser.values)

    return result
