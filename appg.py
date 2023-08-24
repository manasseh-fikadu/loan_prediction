import gradio as gr
import joblib
import pandas as pd
import random

# Load the pipeline
pipeline = joblib.load('pipeline.joblib')
X_test_smaller = joblib.load('X_test_smaller.joblib')

# Define input components
last_pymnt_amnt = gr.inputs.Number(label='Last Payment Amount')
dti = gr.inputs.Number(label='DTI (Debt-to-Income Ratio)')
mo_sin_old_rev_tl_op = gr.inputs.Number(
    label='Months since Oldest Revolving Account')
int_rate = gr.inputs.Number(label='Interest Rate')
revol_util = gr.inputs.Number(label='Revolving Utilization Rate')
bc_open_to_buy = gr.inputs.Number(label='Credit Line Balance')
revol_bal = gr.inputs.Number(label='Revolving Balance')
avg_cur_bal = gr.inputs.Number(label='Average Current Balance')
total_bal_ex_mort = gr.inputs.Number(label='Total Balance Excluding Mortgage')
annual_inc = gr.inputs.Number(label='Annual Income')
loan_amnt = gr.inputs.Number(label='Loan Amount')

def greet():
    return "Hello!"

generate_button = gr.Button(value="Generate Random Row")

# Combine input components
input_components = [last_pymnt_amnt, dti, mo_sin_old_rev_tl_op, int_rate, revol_util,
                    bc_open_to_buy, revol_bal, avg_cur_bal, total_bal_ex_mort,
                    annual_inc, loan_amnt, generate_button]

# Define prediction function
def predict_loan_approval(last_pymnt_amnt, dti, mo_sin_old_rev_tl_op, int_rate, revol_util,
                          bc_open_to_buy, revol_bal, avg_cur_bal, total_bal_ex_mort,
                          annual_inc, loan_amnt, generate_random_row):
    if generate_random_row:
        random_index = random.randint(0, len(X_test_smaller) - 1)
        # Get the row at the random index
        random_row = X_test_smaller.iloc[random_index]
        # Fill input values with random data
        last_pymnt_amnt = random_row['last_pymnt_amnt']
        dti = random_row['dti']
        mo_sin_old_rev_tl_op = random_row['mo_sin_old_rev_tl_op']
        int_rate = random_row['int_rate']
        revol_util = random_row['revol_util']
        bc_open_to_buy = random_row['bc_open_to_buy']
        revol_bal = random_row['revol_bal']
        avg_cur_bal = random_row['avg_cur_bal']
        total_bal_ex_mort = random_row['total_bal_ex_mort']
        annual_inc = random_row['annual_inc']
        loan_amnt = random_row['loan_amnt']

        input_data = {
        'last_pymnt_amnt': last_pymnt_amnt,
        'dti': dti,
        'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op,
        'int_rate': int_rate,
        'revol_util': revol_util,
        'bc_open_to_buy': bc_open_to_buy,
        'revol_bal': revol_bal,
        'avg_cur_bal': avg_cur_bal,
        'total_bal_ex_mort': total_bal_ex_mort,
        'annual_inc': annual_inc,
        'loan_amnt': loan_amnt
        }

        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)
        predict_proba = pipeline.predict_proba(input_df)

        if prediction[0] == 1:
            return f'Fully paid with: {round(predict_proba[0][1] * 100, 2)}% probability \n\nRandomly Generated Row:\n {random_row}'
        else:
            return f'Charged Off with: {round(predict_proba[0][0] * 100, 2)}% probability \nRandomly Generated Row:\n {random_row}'
        
    else:
        input_data = {
            'last_pymnt_amnt': last_pymnt_amnt,
            'dti': dti,
            'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op,
            'int_rate': int_rate,
            'revol_util': revol_util,
            'bc_open_to_buy': bc_open_to_buy,
            'revol_bal': revol_bal,
            'avg_cur_bal': avg_cur_bal,
            'total_bal_ex_mort': total_bal_ex_mort,
            'annual_inc': annual_inc,
            'loan_amnt': loan_amnt
        }

        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)
        predict_proba = pipeline.predict_proba(input_df)

        if prediction[0] == 1:
            return f'Fully paid with {round(predict_proba[0][1] * 100, 2)}% probability'
        else:
            return f'Charged Off with {round(predict_proba[0][0] * 100, 2)}% probability'


# Create the Gradio interface
iface = gr.Interface(
    fn=predict_loan_approval,
    inputs=input_components,
    outputs='text',
    title='Loan Approval Predictor',
    description='If you are using the random row generator, click the button to generate a random row. Then click the Submit button to make a prediction.'
)
iface.launch()
