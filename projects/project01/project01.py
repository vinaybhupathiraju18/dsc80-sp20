
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    cols = grades.columns
    keys = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    assignment_dict = dict.fromkeys(keys)
    assignment_dict['lab'] = [x for x in cols if 'lab' in x and '-' not in x]
    assignment_dict['project'] = [x for x in cols if 'project' in x and '-' not in x and '_' not in x]
    assignment_dict['midterm'] = [x for x in cols if 'Midterm' in x and '-' not in x]
    assignment_dict['final'] = [x for x in cols if 'Final' in x and '-' not in x]
    assignment_dict['disc'] = [x for x in cols if 'disc' in x and '-' not in x]
    assignment_dict['checkpoint'] = [x for x in cols if 'checkpoint' in x and '-' not in x]
    
    return assignment_dict


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def only_proj(df):
    cols = df.columns
    project_cols = [x for x in cols if 'project' in x and '-' not in x and '_' not in x]
    for i in project_cols:
        df[i] = df[i].fillna(0)
    return df[project_cols]

def all_proj_detail(df):
    cols = df.columns
    project_cols = [x for x in cols if 'project' in x and 'check' not in x and 'Late' not in x]
    for i in project_cols:
        df[i] = df[i].fillna(0)
    return df[project_cols]

def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    project = only_proj(grades)
    all_proj = all_proj_detail(grades)
    # create empty dataframe to store project grades to later take the mean from
    scores_df = pd.DataFrame()
    for i in project.columns:
        # gets number of observations to create series of zeros for when there is no FR section
        size = len(all_proj[i])
        fr_score = pd.Series(np.zeros(shape = size))
        # set fr_max to zero in case no FR section
        fr_max = 0
        # gets a series of all projects scores and sets max_pts to max for that project
        proj_pts = all_proj[i]
        max_pts = all_proj[i + ' - Max Points'].max()
        # if there is a FR section, reassign fr variables to series of scores and max value
        if i + '_free_response' in all_proj.columns:
            fr_score = all_proj[i + '_free_response']
            fr_max = all_proj[i + '_free_response - Max Points'].max()
        # add project pts and fr pts. if no fr section, fr_score should be series of 0s and fr_max should be 0
        total_pts = proj_pts + fr_score
        total_max = max_pts + fr_max
        projects_total = total_pts / int(total_max)
        scores_df[i] = projects_total
    
    scores_df['mean'] = scores_df.mean(axis = 1)
    
    return scores_df['mean']


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """ 
    
    # get late labs columns
    cols = grades.columns
    late_labs = [x for x in cols if 'Lateness' in x and 'lab' in x]
    assignment_dict = get_assignment_names(grades)
    labs = assignment_dict['lab']
    
    # get nubmer of labs
    n = len(late_labs)
    incorrect_late_count = []

    # threshhold s.t. labs turned in on time are still considered late
    threshhold = '02:30:00'
    
    # loop through all lab assignment late columns
    for lab in late_labs:
        # get num students that turned in the assingment past the deadline but before threshhold
        num_incorrect_late = len(grades[(grades[lab] < threshhold) & (grades[lab] > '00:00:00')])
        incorrect_late_count.append(num_incorrect_late)
        
        # changing the gradescope bug so that these students aren't penalized in Q4
        # bug_index = grades[(grades[lab] < threshhold) & (grades[lab] > '00:00:00')].index
        # grades[lab][bug_index] = '00:00:00'
        
    result = pd.Series(incorrect_late_count)
    result.index = labs
    return result


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def penalty(x):
    threshold = '02:30:00'
    if x < threshold:
        x = 1.0
    elif threshold < x <= '168:00:00':
        x = 0.9
    elif '168:00:00' < x <= '336:00:00':
        x = 0.8
    elif x > '336:00:00':
        x = 0.5
    return x

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """

    penalties = col.apply(penalty)
    
    return penalties


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
        
    return ...


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    return ...


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """

    return ...

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return ...

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
