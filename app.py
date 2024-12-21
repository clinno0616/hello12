#pip install streamlit pandas plotly openpyxl xlrd numpy
#pip install sqlalchemy
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

def load_excel_file(uploaded_file):
    """載入Excel檔案並返回工作表列表"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names
    except Exception as e:
        st.error(f"無法讀取Excel檔案：{str(e)}")
        return None

def load_sheet_data(uploaded_file, sheet_name):
    """載入指定工作表的數據"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"無法讀取工作表：{str(e)}")
        return None

def filter_dataframe(df, filter_column, selected_values):
    """根據選擇的欄位值過濾數據框"""
    if filter_column and selected_values:
        return df[df[filter_column].isin(selected_values)]
    return df

def create_chart(df, x_column, y_column, chart_type, **chart_params):
    """根據選擇創建對應的圖表"""
    # 計算數值的加總
    if np.issubdtype(df[y_column].dtype, np.number):
        total = df[y_column].sum()
        title_text = f"{y_column} vs {x_column} (總計: {total:,.0f})"
    else:
        title_text = f"{y_column} vs {x_column}"
    try:
        # 設置字體顏色為黑色的基本配置
        layout_config = {
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(t=50, l=50, r=50, b=50),
            'font': dict(color='black'),
            'title': dict(
                text=title_text,
                font=dict(color='black', size=16)
            ),
            'xaxis': dict(
                title=dict(text=x_column, font=dict(color='black')),
                tickfont=dict(color='black'),
                gridcolor='lightgray'
            ),
            'yaxis': dict(
                title=dict(text=y_column, font=dict(color='black')),
                tickfont=dict(color='black'),
                gridcolor='lightgray'
            ),
            'legend': dict(
                font=dict(color='black'),
                bgcolor='rgba(255,255,255,0.8)'
            )
        }

        if chart_type == "折線圖":
            # 使用傳入的參數或預設值
            line_shape = chart_params.get('line_shape', 'linear')
            line_mode = chart_params.get('line_mode', 'lines+markers')
            marker_size = chart_params.get('marker_size', 6)
            line_width = chart_params.get('line_width', 2)
            
            fig = px.line(
                df, 
                x=x_column, 
                y=y_column,
                line_shape=line_shape,  # 折線形狀
                render_mode=line_mode,  # 折線模式
                markers=(True if 'markers' in line_mode else False)
            )
            
            # 更新線條和標記的樣式
            fig.update_traces(
                line=dict(width=line_width),
                marker=dict(size=marker_size)
            )

        elif chart_type == "柱狀圖":
            fig = px.bar(df, x=x_column, y=y_column)
        elif chart_type == "散點圖":
            fig = px.scatter(df, x=x_column, y=y_column)
        elif chart_type == "圓餅圖":
            fig = px.pie(df, values=y_column, names=x_column)
            layout_config.update({
                'title': dict(text=f"{y_column} 分布", font=dict(color='black', size=16)),
            })
        elif chart_type == "長條圖":
            fig = px.bar(df, x=x_column, y=y_column, orientation='h')
        
        # 應用配置
        fig.update_layout(**layout_config)

        # 更新軸線顏色
        if chart_type != "圓餅圖":
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
        return fig
    except Exception as e:
        st.error(f"創建圖表時發生錯誤：{str(e)}")
        return None


# 設置頁面配置
st.set_page_config(
    page_title="Excel資料分析對話系統",
    page_icon="📊",
    layout="wide"
)
def main():
    # 初始化 session state
    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        
    st.title("📊 銷貨資料問答系統")
    
    
    
    # 側邊欄：文件上傳
    with st.sidebar:
        st.header("📂 檔案上傳")
        uploaded_file = st.file_uploader(
            "選擇Excel檔案",
            type=['xlsx', 'xls'],
            help="支援.xlsx和.xls格式"
        )
        
        if uploaded_file is not None:
            # 獲取工作表列表
            sheet_names = load_excel_file(uploaded_file)
            if sheet_names:
                selected_sheet = st.selectbox(
                    "選擇工作表",
                    options=sheet_names,
                    help="選擇要分析的工作表"
                )
                
                # 載入工作表數據
                st.session_state['df'] = load_sheet_data(uploaded_file, selected_sheet)
                if st.session_state['df'] is not None:
                    st.success("數據載入成功！")
                    
                    # 顯示數據基本信息
                    st.markdown("### 📊 數據概覽")
                    st.write(f"行數：{len(st.session_state['df'])}")
                    st.write(f"列數：{len(st.session_state['df'].columns)}")
    
    # 初始化 filtered_data 為 None
    filtered_data = None
    
    # 主要內容區域
    if st.session_state['df'] is not None:
        df = st.session_state['df']

        # 移動對話框到最上方
        #st.header("💬 在此提問:")
        chat_container = st.container()
        # 數據預覽與欄位篩選
        st.header("📋 數據預覽與篩選")
        
        # 欄位篩選區域
        with st.expander("欄位篩選設置", expanded=True):
            # 建立多列布局用於選擇欄位和值
            num_cols = 4  # 每行顯示的篩選器數量
            cols = st.columns(num_cols)
            # 初始化篩選條件
            filter_conditions = {}
            
            # 為每個欄位創建篩選器
            for idx, col in enumerate(df.columns):
                current_col = cols[idx % num_cols]
                if col=='銷貨數量':
                    break
                with current_col:
                    # 獲取列的數據類型
                    dtype = df[col].dtype
                    
                    # 創建欄位篩選器
                    st.markdown(f"**{col}**")
                    if dtype == 'object' or dtype == 'string':
                        unique_values = [str(x) for x in df[col].unique() if pd.notna(x)]
                        selected_values = st.multiselect(
                            f"選擇 {col} 的值",
                            options=unique_values,
                            default=None,
                            key=f"filter_{col}"
                        )
                        if selected_values:
                            filter_conditions[col] = selected_values
                            
                    elif np.issubdtype(dtype, np.number):
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        range_values = st.slider(
                            f"選擇 {col} 的範圍",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filter_{col}"
                        )
                        if range_values != (min_val, max_val):
                            filter_conditions[col] = range_values
                            
                    elif np.issubdtype(dtype, np.datetime64):
                        min_date = df[col].min()
                        max_date = df[col].max()
                        start_date = st.date_input(
                            f"選擇 {col} 起始日期",
                            value=min_date,
                            key=f"filter_{col}_start"
                        )
                        end_date = st.date_input(
                            f"選擇 {col} 結束日期",
                            value=max_date,
                            key=f"filter_{col}_end"
                        )
                        if start_date != min_date or end_date != max_date:
                            filter_conditions[col] = (start_date, end_date)
        
        # 應用篩選條件
        filtered_data = df.copy()
        for col, condition in filter_conditions.items():
            if isinstance(condition, list):  # 文字類型的多選
                filtered_data = filtered_data[filtered_data[col].astype(str).isin(condition)]
            elif isinstance(condition, tuple):  # 數值或日期範圍
                if np.issubdtype(df[col].dtype, np.datetime64):
                    filtered_data = filtered_data[
                        (filtered_data[col].dt.date >= condition[0]) &
                        (filtered_data[col].dt.date <= condition[1])
                    ]
                else:
                    filtered_data = filtered_data[
                        (filtered_data[col] >= condition[0]) &
                        (filtered_data[col] <= condition[1])
                    ]
        
        # 更新 session state 中的 filtered_data
        st.session_state['filtered_df'] = filtered_data

        # 視覺化設置
        st.header("📈 視覺化設置")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 設定X軸預設值為年月
            default_x = "年月" if "年月" in filtered_data.columns else filtered_data.columns[0]
            x_column = st.selectbox(
                "選擇X軸",
                options=filtered_data.columns,
                index=list(filtered_data.columns).index(default_x),
                help="選擇X軸數據"
            )
        
        with col2:
            # 設定Y軸預設值為銷貨金額
            default_y = "銷貨金額" if "銷貨金額" in filtered_data.columns else filtered_data.columns[0]
            y_column = st.selectbox(
                "選擇Y軸",
                options=filtered_data.columns,
                index=list(filtered_data.columns).index(default_y),
                help="選擇Y軸數據"
            )
        
        with col3:
            chart_type = st.selectbox(
                "選擇圖表類型",
                options=["折線圖", "柱狀圖", "散點圖", "圓餅圖", "長條圖"],
                index=1,  # 設定預設值為柱狀圖（index=1）
                help="選擇要展示的圖表類型"
            )
        
        # 折線圖特定設置
        chart_params = {}
        if chart_type == "折線圖":
            st.markdown("### 折線圖設置")
            line_col1, line_col2 = st.columns(2)
            
            with line_col1:
                line_shape = st.selectbox(
                    "選擇線條形狀",
                    options=["linear", "spline", "hv", "vh", "hvh", "vhv"],
                    help="""
                    linear: 直線連接
                    spline: 平滑曲線
                    hv: 先水平後垂直
                    vh: 先垂直後水平
                    hvh: 水平-垂直-水平
                    vhv: 垂直-水平-垂直
                    """
                )
                chart_params['line_shape'] = line_shape
            
            with line_col2:
                line_mode = st.selectbox(
                    "選擇線條模式",
                    options=["lines", "lines+markers", "markers"],
                    help="""
                    lines: 只顯示線條
                    lines+markers: 顯示線條和數據點
                    markers: 只顯示數據點
                    """
                )
                chart_params['line_mode'] = line_mode

        # 創建並顯示圖表
        if x_column and y_column:
            fig = create_chart(filtered_data, x_column, y_column, chart_type, **chart_params)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # 下載圖表選項
                st.download_button(
                    label="下載圖表",
                    data=fig.to_html(),
                    file_name=f"{chart_type}_{x_column}_{y_column}.html",
                    mime="text/html"
                )

       # 在對話框中使用最新的 filtered_data
        with chat_container:
            # 初始化 session state 用於追蹤提交狀態
            if "submit_state" not in st.session_state:
                st.session_state.submit_state = False
            
            # 在表單之後顯示對話歷史
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**👤 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")    
            # 創建一個表單來處理輸入
            with st.form(key='chat_form'):
                user_input = st.text_input("請輸入您的問題", 
                    placeholder="例如：銷貨金額最高的是哪一筆資料？")
                submit_button = st.form_submit_button("送出")
            
            if submit_button and user_input and st.session_state['filtered_df'] is not None:
                # 使用最新的 filtered_data
                current_data = st.session_state['filtered_df'].to_string()
                
                try:
                    # 初始化 Gemini 模型
                    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                    #model = genai.GenerativeModel("gemini-1.5-pro-latest")
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    
                    # 構建提示詞
                    prompt = f"""基於以下數據回答問題。數據內容如下：
                    
                    {current_data}
                    
                    問題：{user_input}
                    
                    請提供詳細的分析和解答。如果問題涉及數值計算，請說明計算過程。使用者非資訊工程背景，不要提供任何與程式相關的內容。"""
                    
                    # 發送請求給模型
                    response = model.generate_content(prompt)
                    
                    # 儲存對話歷史
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"分析過程發生錯誤：{str(e)}")

                    
            # 清除對話按鈕
            if st.button("清除對話歷史"):
                st.session_state.chat_history = []
                st.rerun()
        
        # 如果有過濾後的數據，顯示數據編輯器
        if filtered_data is not None:
            st.markdown("### 篩選後的數據")
            display_df = st.data_editor(
                filtered_data,
                use_container_width=True,
                num_rows="dynamic",
                height=400,
                hide_index=False,
                disabled=True,
                key="data_editor"
            )
            
            # 顯示當前篩選後的數據統計
            total_rows = len(display_df)
            total_cols = len(display_df.columns)
            st.caption(f"當前顯示行數: {total_rows} | 總列數: {total_cols}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"系統錯誤：{str(e)}")
        st.error("請重新加載頁面或聯繫系統管理員")