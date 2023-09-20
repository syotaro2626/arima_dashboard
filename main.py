import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pmdarima as pm

def ARIMA(df, seasonal):
    #dfをtrainとtestに分ける
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # モデル構築（Auto ARIMA）
    arima_model = pm.auto_arima(train_df.values, 
                                seasonal=True,
                                m=seasonal,
                                trace=True,
                                n_jobs=-1,
                                maxiter=10,
                                verbose=0)
    test_pred, test_pred_ci = arima_model.predict(n_periods=test_df.shape[0], 
                                                  return_conf_int=True)
    
    #予測結果のグラフを表示
    fig, ax = plt.subplots()
    ax.plot(train_df.index, train_df.values, label="actual(train dataset)", color='tab:blue')
    ax.plot(test_df.index, test_df.values, label="actual(test dataset)", color='tab:blue')
    ax.plot(test_df.index, test_pred, label="auto ARIMA", color="tab:red")
    ax.fill_between(test_df.index, test_pred_ci[:, 0], test_pred_ci[:, 1], color='tab:red', alpha=.2)
    ax.legend()
    plt.xticks(rotation=45)
    plt.title('prediction by ARIMA')
    st.pyplot(fig)
                    

#WEBアプリ部分
st.title('時系列データ予測アプリ')
st.caption('投入したデータをARIMAを使って自動で予測します')

datafile = st.file_uploader("upload file", type='csv')
if datafile:
    df = pd.read_csv(datafile)
    df.reset_index(inplace= True)
    cols = df.columns
    st.text(f'レコード数：{len(df)}')
    st.dataframe(df)

    with st.form(key='form'):
        x_axis = st.selectbox('x軸    ※日付型のカラムを選択してください。', cols)
        y_axis = st.selectbox('y軸', cols)
        seasonal = st.number_input('季節性', value=12)

        submit_btn = st.form_submit_button('submit')
        if submit_btn:
            df = df[[x_axis, y_axis]]
            df[x_axis] = pd.to_datetime(df[x_axis])
            df = df.set_index(x_axis)

            ARIMA(df, seasonal)