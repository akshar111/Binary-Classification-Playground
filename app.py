import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score
import traceback

accuracy = 0
precision = 0
recall = 0
f1 = 0

if 'acc' not in st.session_state:
    st.session_state.acc = 0
if 'prec' not in st.session_state:
    st.session_state.prec = 0
if 'recall' not in st.session_state:
    st.session_state.recall = 0
if 'f1' not in st.session_state:
    st.session_state.f1 = 0

# Function for loading and preprocessing data (caching for efficiency)
@st.cache_data(persist=True)
def load_data(data_path="./mushrooms.csv"):
    try:
        data = pd.read_csv(data_path)
        labelencoder = LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        X = data.drop(columns=['type'])
        # Assuming target variable is named 'type'
        y = data['type']
        return X, y
    except FileNotFoundError:
        st.error(f"Data file '{data_path}' not found. Please ensure it exists.")
        return None, None  # Return None to prevent errors downstream

# Function for splitting data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

# Function for training and evaluating the selected classifier
def train_evaluate_model(classifier_type, X_train, X_test, y_train, y_test):
    model = None  # Initialize model to None
    try:
        if classifier_type == "SVM":
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM', help="The maximum value for C is 10, and the step size is 0.01.")
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel', help="The 'Kernel' parameter specifies the kernel type to be used in the algorithm. 'rbf' stands for radial basis function.")
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma', help="The 'Gamma' parameter defines how far the influence of a single training example reaches, with low values meaning 'far' and high values meaning 'close'.")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

        elif classifier_type == "Logistic Regression":
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR', help="The maximum value for C is 10, and the step size is 0.01.")
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter', help="The 'Maximum number of iterations' parameter specifies the maximum number of iterations taken for the solvers to converge.")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)

        elif classifier_type == "Random Forest":
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                                   key='n_estimators', help="This parameter specifies the number of trees in the random forest model.")
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth', help="This parameter specifies the maximum depth of each decision tree in the random forest.")
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap', help="This parameter specifies whether bootstrap samples are used when building trees in the random forest model.")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

        return model
    except Exception as e:  # Catch general exceptions
        st.error(f"An error occurred in train_evaluate_model function: {e}")
def plot_model(model, X_train, X_test, y_train, y_test):
    class_names = ['edible', 'poisonous']
    global accuracy, precision, recall, f1
    classify_btn = st.sidebar.button("Classify", key='classify', use_container_width=True, type="primary")
    if not classify_btn:
        st.info("Please adjust the parameters on the left and click the 'Classify' button to start the classification.")
    try:
        if classify_btn:
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            y_pred = model.predict(X_test)
            precision = precision_score(y_test, y_pred, labels=class_names, zero_division=1)
            recall = recall_score(y_test, y_pred, labels=class_names)
            f1 = f1_score(y_test, y_pred, labels=class_names)

            # st.session_state.acc = accuracy
            deffacc = accuracy - st.session_state.acc
            deffprec = precision - st.session_state.prec
            deffrecall = recall - st.session_state.recall
            defff1 = f1 - st.session_state.f1

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label='Accuracy', value=f"{accuracy:.2f}", delta=f"{deffacc:.2f}", delta_color='normal')
            with col2:
                st.metric(label='Precision', value=f"{precision:.2f}", delta=f"{deffprec:.2f}", delta_color='normal')
            with col3:
                st.metric(label='Recall', value=f"{recall:.2f}", delta=f"{deffrecall:.2f}", delta_color='normal')
            with col4:
                st.metric(label='F1 Score', value=f"{f1:.2f}", delta=f"{defff1:.2f}", delta_color='normal')

            st.write("---")

            plot_confusion_matrix(y_test, y_pred, class_names)
            col1, col2 = st.columns(2)
            with col1:
                plot_roc_curve(model, X_test, y_test)
            with col2:
                plot_precision_recall_curve(model, X_test, y_test)

            # Update session state here
            if 'acc' not in st.session_state:
                st.session_state.acc = accuracy
            else:
                st.session_state.acc = accuracy
            if 'prec' not in st.session_state:
                st.session_state.prec = precision
            else:
                st.session_state.prec = precision
            if 'recall' not in st.session_state:
                st.session_state.recall = recall
            else:
                st.session_state.recall = recall
            if 'f1' not in st.session_state:
                st.session_state.f1 = f1
            else:
                st.session_state.f1 = f1

    except Exception as e:  # Catch general exceptions
        traceback_str = traceback.format_exc()
        st.error(f"An error occurred in plot_model function: {e} \n{traceback_str}")

# Function for plotting confusion matrix using Plotly (interactive)
def plot_confusion_matrix(y_test, y_pred, class_names):
    matrix = confusion_matrix(y_test, y_pred)
    fig = px.imshow(matrix,
                    labels=dict(x="Predicted Label", y="True Label", color="Counts"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale='Viridis')
    fig.update_layout(title='Confusion Matrix', title_x=0.4)
    st.plotly_chart(fig)

# Function for plotting ROC curve using Plotly
def plot_roc_curve(model, X_test, y_test):
    try:
        if hasattr(model, "predict_proba"):
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            fig = px.area(x=fpr, y=tpr,
                          labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')

            # Center the plot within the column
            fig.update_layout(title='ROC Curve', title_x=0.4, width=450, height=450)

            st.plotly_chart(fig)
        else:
            st.warning("ROC Curve cannot be plotted as the model does not support probability estimation.")
    except Exception as e:
        st.error(f"An error occurred in plot_roc_curve function: {e}")

# Function for plotting precision-recall curve using Plotly
def plot_precision_recall_curve(model, X_test, y_test):
    try:
        if hasattr(model, "predict_proba"):
            precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
            fig = px.area(x=recall, y=precision,
                          labels=dict(x='Recall', y='Precision'))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            fig.update_layout(title='Precision-Recall Curve', title_x=0.3, width=450, height=450)
            st.plotly_chart(fig)
        else:
            st.warning("Precision-Recall Curve cannot be plotted as the model does not support probability estimation.")
    except Exception as e:
        st.error(f"An error occurred in plot_precision_recall_curve function: {e}")

if __name__ == '__main__':

    st.subheader("Binary Classification Playground")
    st.sidebar.title("Adjust Parameters")

    try:
        X, y = load_data()
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = split_data(X, y)

        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Mushroom Data Set (Classification)")
            data_path = "./mushrooms.csv"
            data = pd.read_csv(data_path)
            st.write(data)

            st.markdown(
                "This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
                "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
                "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

        classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"), help="Select the classifier you want to use for classification.")

        if classifier:  # Handle case where no classifier is chosen initially
            model = train_evaluate_model(classifier, X_train, X_test, y_train, y_test)
            plot_model(model, X_train, X_test, y_train, y_test)

    except Exception as e:  # Catch general exceptions
        st.error(f"An error occurred in main function: {e}")


