import streamlit as st
from streamlit.components.v1 import html

import pandas as pd
import altair as alt

from math import cos, sin, atan, pi, degrees
from numpy.random import randint as RndInt, seed as Seed
from numpy import inf as Inf, isclose, array
from numpy.linalg import norm as Norm

# config, util, declr
if True:
    import warnings
    warnings.filterwarnings('ignore')

    def page_config():
        st.set_page_config(page_title='Shortest Path Problem',
                        layout='wide')

    page_config()

    color_arc_sel = ['cornflowerblue', 'mediumblue', 'darkblue'][1]
    color_arc_des = ['gray', 'lightgray', 'darkgray'][-1]
    color_node = ['blue', 'cornflowerblue', 'green'][-1]
    color_node_label = 'white'
    color_node_descr = 'gray'
    color_node_text  = 'lightgray'
    color_arc_cost = ['orange', 'black'][0]

    js_text_areas = """\
        <script>
        const TA = window.parent.document.querySelectorAll('textarea[type="textarea"]');
        TA.forEach(t => {
            t.spellcheck = false;
            t.style.setProperty('font-family', 'monospace')
            t.style.setProperty('font-size',   'smaller')
            }
        );
        </script>
    """

    def vector_delta(x1, y1, x2, y2, dx, dy):
        L = Norm(((x2 - x1), (y2 - y1)))
        u = ((x2 - x1)/L, (y2 - y1)/L)
        a = atan(Inf if isclose(u[0], 0, 0.00001, 0.00001) else u[1]/u[0])
        d = Norm((abs(cos(a))*dx, abs(sin(a))*dy))
        return (x2 - d*u[0], y2 - d*u[1])


    def vector_ortho(x1, y1, x2, y2):
        L = Norm((x2 - x1, y2 - y1))
        u = ((x2 - x1)/L, (y2 - y1)/L)
        return (u[1], -u[0])


    def vector_angle(x1, y1, x2, y2):
        dx,dy = x2 - x1, y2 - y1
        a = atan(Inf if isclose(dx, 0, 0.00001, 0.00001) else dy/dx)
        a += pi if dx < 0 else 0
        # a = acos(Inf if isclose(dx, 0, 0.00001, 0.00001) else dy/dx)
        return a

    example_nodes = '\n'.join([r.strip() for r in """\
        A 15 50 Toronto
        B 25 40 New_York
        C 10 30 Mexico_City
        D 25 20 Sao_Paolo
        E 35 35 Madrid
        F 40 45 London
        G 50 50 Berlin
        H 50 35 Athens
        I 55 40 İstanbul
        J 60 30 Delhi
        K 70 35 Beijing
        L 75 45 Tokyo
    """.splitlines() if len(r)])
    example_adj = '\n'.join([r.strip() for r in """\
        A : B F C     : 10 20 80
        B : A C D E F : 30 40 60 50 70
        C : A B D     : 20 50 90
        D : B C J     : 40 30 80
        E : B F       : 60 20
        F : A B E G H : 60 50 20 30 40
        G : F I L     : 40 70 60
        H : F I J     : 30 50 30
        I : G H J     : 40 20 50
        J : D K H I   : 90 40 30 40
        K : J L       : 50 30
        L : G K       : 70 40
    """.splitlines() if len(r)])
    example_walks = '\n'.join([r.strip() for r in """\
        A-B A-C A-D
    """.splitlines()])
    example_colors = '\n'.join([r.strip() for r in """\
        red : B
        cornflowerblue : A D
        # orange : C E
    """.splitlines() if len(r)])
    example_text = '\n'.join([r.strip() for r in """\
        A : [* 0]
        B : [A 100]
        C : [- ∞]
        D : [- ∞]
        E : [- ∞]
    """.splitlines() if len(r)])


# sidebar
if True:
    sidebar_R = []

    # st.sidebar.write(f"Shortest Path Problem")

    st.sidebar.write("")

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        t_graph = st.text_input('Graph', value='720 480', label_visibility='collapsed')
        s_graph = [int(s) for s in t_graph.split()]
    with sidebar_R[-1][1]:
        p_graph = st.checkbox('Graph', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        s_nodes  = st.number_input('Node Size', value=500, step=50, label_visibility='collapsed')
    with sidebar_R[-1][1]:
        p_nodes  = st.checkbox('Nodes', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        s_labels  = st.number_input('Labels', value=16, step=2, label_visibility='collapsed')
    with sidebar_R[-1][1]:
        p_labels  = st.checkbox('Labels', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        s_desc = st.number_input('Descr', value=16, step=2, label_visibility='collapsed')
    with sidebar_R[-1][1]:
        p_desc = st.checkbox('Descr', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        s_arcs = st.number_input('Arc Width', value=4, step=1, label_visibility='collapsed')
    with sidebar_R[-1][1]:
        p_arcs = st.checkbox('Arcs', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        s_arrows = st.number_input('Arr Sz', value=12, step=1, label_visibility='collapsed')
    with sidebar_R[-1][1]:
        p_arrows = st.checkbox('Arc Arr', value=True)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        t_costs = st.text_input('Cost Size + Offset', value='14 1.3 1.0', label_visibility='collapsed')
        s_costs = [float(s) for s in t_costs.split()]
    with sidebar_R[-1][1]:
        p_costs = st.checkbox('Costs', value=False)

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        t_text = st.text_input('Text Size + Offset', value='14 12 10', label_visibility='collapsed')
        s_text = [float(s) for s in t_text.split()]
    with sidebar_R[-1][1]:
        p_text = st.checkbox('Text', value=False)

    st.sidebar.write("")

    # o_option = st.sidebar.selectbox('Arc Sel', ('Textbox', 'Checkbox'))
    o_option = 'Textbox'


    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        w_arc_cols = st.text_input('Cols Rt',  value='4 10')
        r_arc_cols = [int(c) for c in w_arc_cols.split()]
    with sidebar_R[-1][1]:
        n_arc_cols = 5

    sidebar_R.append(st.sidebar.columns([1, 1], vertical_alignment="bottom"))
    with sidebar_R[-1][0]:
        # s_Abx = st.number_input('A base x', value=2.0, step=0.1)
        # s_Aby = st.number_input('A base y', value=2.0, step=0.1)
        t_A_base = st.text_input('A base ', value="2.4 2.4")
        s_Abx,s_Aby = (float(b) for b in t_A_base.split())
    with sidebar_R[-1][1]:
        # s_Ahx = st.number_input('A head x', value=1.0, step=0.1)
        # s_Ahy = st.number_input('A head y', value=1.0, step=0.1)
        t_A_head = st.text_input('A head', value="1.1 1.1")
        s_Ahx,s_Ahy = (float(b) for b in t_A_head.split())

    st.sidebar.write("")

    p_arcs_data   = st.sidebar.checkbox('Arcs Data', value=False)
    p_nodes_data  = st.sidebar.checkbox('Nodes Data', value=False)
    p_arcarr_data = st.sidebar.checkbox('Arc Arw Data', value=False)


# main area
if True:
    # st.subheader('Network Tool')
    st.markdown('**Network Tool**')

    gr_cols = st.columns(r_arc_cols, vertical_alignment='top')

    with gr_cols[0]:
        tab_ins_nodes, tab_ins_arcs, tab_arc_walks, tab_node_colors, tab_node_text = \
            st.tabs(['Nodes', 'Arcs', 'Walks', 'Colors', 'Text'])

        # Problem Instance: Nodes
        with tab_ins_nodes:
            ins_nodes_txt = st.text_area('Coords & Descriptions', value=example_nodes, height=320, help='lbl x y decr')
            ins_Nodes = {}
            for L in ins_nodes_txt.splitlines():
                K = L.split()
                ins_Nodes[K[0]] = [int(K[1]), int(K[2])]
                ins_Nodes[K[0]].append(K[3].replace('_', ' ') if len(K) > 3 else '')

        # Problem Instance: Arcs
        with tab_ins_arcs:
            ins_arcs_txt  = st.text_area('Adjacency & Costs', value=example_adj, height=320, help='head : tail(s) : cost(s)')
            ins_Arcs = {}
            for L in ins_arcs_txt.splitlines():
                h = L.split(':')[0].strip()
                for t,c in zip(L.split(':')[1].split(),L.split(':')[2].split()):
                    ins_Arcs[f"{h}-{t.strip()}"] = [h, t.strip(), int(c)]

        d_Nodes = pd.DataFrame(data={
            'n'           : [k     for k in ins_Nodes.keys()],
            'description' : [v[2]  for _,v in ins_Nodes.items()],
            'x'           : [v[0]  for _,v in ins_Nodes.items()],
            'y'           : [v[1]  for _,v in ins_Nodes.items()],
            'color'       : [color_node for _ in ins_Nodes.items()],
            'text'        : [''         for _ in ins_Nodes.items()],
            })
        d_Nodes.set_index(keys='n', inplace=True)
        d_Nodes['label'] = d_Nodes.index.astype(str)
        Nodes = list(d_Nodes.index)

        Arcs = { (v[0],v[1]) : v[2]  for a,v in ins_Arcs.items() }
        Arc_Adj  = { n : [v[1] for a,v in ins_Arcs.items() if v[0]==n] for n in Nodes }
        Arc_Cost = { n : [v[2] for a,v in ins_Arcs.items() if v[0]==n] for n in Nodes }
        
        d_Arcs = pd.DataFrame(dict(
                    arc = [f'{h}-{t}'         for h,t in Arcs.keys()],
                    xh  = [d_Nodes.loc[h,'x'] for h,t in Arcs.keys()],
                    yh  = [d_Nodes.loc[h,'y'] for h,t in Arcs.keys()],
                    xt  = [d_Nodes.loc[t,'x'] for h,t in Arcs.keys()],
                    yt  = [d_Nodes.loc[t,'y'] for h,t in Arcs.keys()],
                    sel = [0                  for _   in Arcs.keys()],
                    c   = [c                  for (h,t),c in Arcs.items()],
                    tt  = [f"{d_Nodes.loc[h,'description']} - {d_Nodes.loc[t,'description']}" for h,t in Arcs.keys()],
                ))
        d_Arcs.set_index('arc', inplace=True)

        # Arc Walks
        with tab_arc_walks:
            arc_sel = {}
            valid_input = True

            if p_graph and o_option == 'Textbox':
                arc_sel_txt = st.text_area('Arcs, Paths, Walks', value=example_walks, height=120, help='h-t a-b-c-d-f')
                Walks = arc_sel_txt.split()
                for W in Walks:
                    for h,t in zip(W.split('-')[:-1],W.split('-')[1:]):
                        if (h,t) in Arcs:
                            if h not in arc_sel:
                                arc_sel[h] = {}
                            arc_sel[h][t] = True
                        else:
                            st.write((h,t))
                            valid_input = False

            if p_graph and o_option == 'Checkbox':
                ap_cols = []
                ap_cols.append(st.columns([1 for _ in range(n_arc_cols)], vertical_alignment="top"))
                i = 0
                for h in Arc_Adj.keys():
                    ap_cols[-1][i].write('')
                    ap_cols[-1][i].write(f'**Adj( {h} )**')
                    arc_sel[h] = {}
                    for t in Arc_Adj[h]:
                        with ap_cols[-1][i]:
                            arc_sel[h][t] = st.checkbox(label=f"{h}-{t}", key=f"{h}-{t}")
                    i += 1
                    if i == n_arc_cols:
                        ap_cols.append(st.columns([1 for _ in range(n_arc_cols)], vertical_alignment="top"))
                        i = 0

            if p_graph and o_option in ('Checkbox', 'Textbox'):
                Total_Cost = 0
                for h in arc_sel.keys():
                    for t in arc_sel[h]:
                        d_Arcs.loc[f"{h}-{t}", "sel"] = 1 if arc_sel[h][t] else 0
                        Total_Cost += d_Arcs.loc[f'{h}-{t}', 'c'] if arc_sel[h][t] else 0
            else:
                Total_Cost = None

            Total_Cost = Total_Cost if valid_input else '<Invalid Input>'
            if Total_Cost is not None:
                st.markdown(f"Total Cost = {Total_Cost}")
                # st.text_input('**Total Cost**', value=f'{Total_Cost}', disabled=True)

            # for o,d in ((1,3), (3,2), (2,4), (4,1)):
            #     st.write(f"{o}-{d} {degrees(vector_angle(d_Nodes.loc[o].x, d_Nodes.loc[o].y, d_Nodes.loc[d].x, d_Nodes.loc[d].y)):.2f}")

        # Node Colors
        with tab_node_colors:
            node_col_txt = st.text_area('Colors', value=example_colors, height=120, help='color : node(s)')
            node_colors = {}
            if len(node_col_txt):
                for L in node_col_txt.splitlines():
                    K = L.split(':')
                    node_colors[K[0].strip()] = K[1].split()
                for c,N in node_colors.items():
                    if c[0] != "#":
                        d_Nodes.loc[N, 'color'] = c

        # Node Text
        with tab_node_text:
            node_text_txt = st.text_area('Text', value=example_text, height=120, help='node : text')
            node_text = {}
            if len(node_text_txt):
                for L in node_text_txt.splitlines():
                    K = L.split(':')
                    node_text[K[0].strip()] = K[1].strip()
                for n,T in node_text.items():
                    d_Nodes.loc[n, 'text'] = T
                # st.write(str(node_text))


# Arc Drawing
if True:
    # TODO:
    #   use explicit column names (too cryptic xt2, xt3, xe, a)
    #   use p_arrows to assign tail coordinates
    d_Arcs['xt2'] = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_Abx, s_Aby)[0], axis=1)
    d_Arcs['yt2'] = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_Abx, s_Aby)[1], axis=1)
    d_Arcs['xt3'] = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_Ahx, s_Ahy)[0], axis=1)
    d_Arcs['yt3'] = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_Ahx, s_Ahy)[1], axis=1)
    d_Arcs['xc']  = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_costs[1]+s_Abx, s_costs[2]+s_Aby)[0], axis=1)
    d_Arcs['yc']  = d_Arcs.apply(lambda R : vector_delta(R.xh, R.yh, R.xt, R.yt, s_costs[1]+s_Abx, s_costs[2]+s_Aby)[1], axis=1)
    d_Arcs['xcd'] = d_Arcs.apply(lambda R : vector_ortho(R.xh, R.yh, R.xt, R.yt)[0], axis=1)
    d_Arcs['ycd'] = d_Arcs.apply(lambda R : vector_ortho(R.xh, R.yh, R.xt, R.yt)[1], axis=1)

    d_Arcs['a']   = d_Arcs.apply(lambda R : round(vector_angle(R.xh, R.yh, R.xt, R.yt), 3), axis=1)
    # d_Arcs['xh']  = d_Arcs.xh  + 0.5*d_Arcs.xcd
    # d_Arcs['xt']  = d_Arcs.xt  + 0.5*d_Arcs.xcd
    # d_Arcs['yh']  = d_Arcs.yh  + 0.5*d_Arcs.ycd
    # d_Arcs['yt']  = d_Arcs.yt  + 0.5*d_Arcs.ycd
    # d_Arcs['xt2'] = d_Arcs.xt2 + 0.5*d_Arcs.xcd
    # d_Arcs['yt2'] = d_Arcs.yt2 + 0.5*d_Arcs.ycd
    # d_Arcs['xt3'] = d_Arcs.xt3 + 0.5*d_Arcs.xcd
    # d_Arcs['yt3'] = d_Arcs.yt3 + 0.5*d_Arcs.ycd

    d_Arcs.loc[d_Arcs.sel == 0, 'xe'] = d_Arcs.loc[d_Arcs.sel == 0]['xt' ]
    d_Arcs.loc[d_Arcs.sel == 0, 'ye'] = d_Arcs.loc[d_Arcs.sel == 0]['yt' ]
    d_Arcs.loc[d_Arcs.sel == 1, 'xe'] = d_Arcs.loc[d_Arcs.sel == 1]['xt2']
    d_Arcs.loc[d_Arcs.sel == 1, 'ye'] = d_Arcs.loc[d_Arcs.sel == 1]['yt2']
    # d_Arcs.loc[d_Arcs.sel == 0, 'xe'] = d_Arcs.loc[d_Arcs.sel == 0]['xt' ] + 0.5*d_Arcs[d_Arcs.sel == 0]['xcd']
    # d_Arcs.loc[d_Arcs.sel == 0, 'ye'] = d_Arcs.loc[d_Arcs.sel == 0]['yt' ] + 0.5*d_Arcs[d_Arcs.sel == 0]['ycd']
    # d_Arcs.loc[d_Arcs.sel == 1, 'xe'] = d_Arcs.loc[d_Arcs.sel == 1]['xt2'] + 0.5*d_Arcs[d_Arcs.sel == 1]['xcd']
    # d_Arcs.loc[d_Arcs.sel == 1, 'ye'] = d_Arcs.loc[d_Arcs.sel == 1]['yt2'] + 0.5*d_Arcs[d_Arcs.sel == 1]['ycd']



# Arc Arrows dataframe
if True:
    d_head = d_Arcs[['sel', 'xt3', 'yt3', 'tt', 'xcd', 'ycd']].rename(columns={'xt3':'xa', 'yt3':'ya'})
    d_base = d_Arcs[['sel', 'xt2', 'yt2', 'tt', 'xcd', 'ycd']].rename(columns={'xt2':'xa', 'yt2':'ya'})
    d_ArcArr = pd.concat((d_base.assign(w=s_arrows), 
                          d_head.assign(w=0.1)))
    # d_ArcArr['xa'] = d_ArcArr['xa'] + 0.5*d_ArcArr.xcd
    # d_ArcArr['ya'] = d_ArcArr['ya'] + 0.5*d_ArcArr.ycd
    d_ArcArr.drop(columns=['xcd', 'ycd'], inplace=True)
    d_ArcArr.reset_index(names='arc', inplace=True)


# dataframe dumps
if True:
    if p_arcs_data:
        st.dataframe(d_Arcs,   hide_index=False)
    if p_nodes_data:
        st.dataframe(d_Nodes,  hide_index=False)
    if p_arcarr_data:
        st.dataframe(d_ArcArr, hide_index=False)


# chart object
if True:
    # common coordinates
    a_Base = alt.Chart(d_Nodes) \
    .encode(
        x = alt.X('x:Q', title=''),
        y = alt.Y('y:Q', title=''),
    )

    # nodes
    a_Nodes = a_Base.mark_circle() \
    .encode(
        x       = alt.X('x:Q', title='').scale(zero=False).axis(labels=False),
        y       = alt.Y('y:Q', title='').scale(zero=False).axis(labels=False),
        size    = alt.value(s_nodes if p_nodes else 0),
        color   = alt.Color('color').scale(None),
        opacity = alt.value(1.0),
        tooltip = 'description:N',
    )


    # node labels
    a_Labels = a_Base.mark_text(
        align      = 'center',
        baseline   = 'middle',
        dx         = 0,
        dy         = 0,
        fontSize   = s_labels if p_labels else 0,
        fontWeight = 'bold',
    ).encode(
        color      = alt.value(color_node_label),
        text       = 'label:N',
        tooltip    = 'description:N',
    )
    # a_Labels = a_Base.mark_text(
    #     dx         = 0,
    #     dy         = 0,
    #     align      = 'center',
    #     baseline   = 'middle',
    #     color      = color_node_label,
    #     fontSize   = s_labels if p_labels else 0,
    #     fontWeight = 'bold',
    # ).encode(
    #     text    = 'label:N',
    #     tooltip = 'description:N',
    # )


    # arc costs
    a_Costs = alt.Chart(d_Arcs) \
    .mark_text(
        dx   = alt.expr(-s_costs[0] * alt.datum.xcd),
        dy   = alt.expr(s_costs[0] * alt.datum.ycd),
        align      = 'center',
        baseline   = 'middle',
        color      = color_arc_cost,
        fontSize   = s_costs[0] if p_costs else 0,
        fontWeight = 'normal',
    ).encode(
        x    = alt.X('xc', type='quantitative', title=''),
        y    = alt.Y('yc', type='quantitative', title=''),
        text = 'c:Q',
        opacity  = alt.condition((alt.datum.sel) == 1, alt.value(1.0), alt.value(0.2)),
    )


    # node descriptions
    a_Descriptions = a_Base.mark_text(
        dx       = 12,
        dy       = 10,
        align    = 'left',
        baseline = 'middle',
        color    = color_node_descr,
        fontSize = s_desc if p_desc else 0,
        fontWeight = 'normal',
        tooltip  = '',
    ).encode(
        text='description:N'
    )


    # node text
    a_Text = a_Base.mark_text(
        dx       = s_text[1],
        dy       = s_text[2],
        align    = 'left',
        baseline = 'middle',
        color    = color_node_text,
        fontSize = s_text[0] if p_text else 0,
        fontWeight = 'normal',
        tooltip  = '',
    ).encode(
        text='text:N'
    )


    # arcs (unselected + selected)
    a_Arcs = alt.Chart(d_Arcs).mark_rule(
        strokeWidth = s_arcs if p_arcs else 0,
    ).encode(
        x       = alt.X('xh', type='quantitative', title=''),
        y       = alt.Y('yh', type='quantitative', title=''),
        x2      = 'xe:Q',
        y2      = 'ye:Q',
        color   = alt.condition((alt.datum.sel) == 1, alt.value(color_arc_sel), alt.value(color_arc_des)),
        opacity = alt.condition((alt.datum.sel) == 1, alt.value(1.0), alt.value(0.1)),
        tooltip = 'tt:N',
    )


    # arc arrows
    a_Arrows = alt.Chart(d_ArcArr).mark_trail() \
    .transform_filter(
        (alt.datum.sel) != 11
    ).encode(
        detail  = 'arc:N',
        x       = alt.X('xa:Q', title=''),
        y       = alt.Y('ya:Q', title=''),
        size    = alt.Size('w:Q', scale=alt.Scale(range=[0, s_arrows]), legend=None),
        color   = alt.condition((alt.datum.sel) == 1, alt.value(color_arc_sel), alt.value(color_arc_des)),
        opacity = alt.condition((alt.datum.sel) == 1, alt.value(1.0 if p_arrows else 0.0), alt.value(0.0)),
        tooltip = 'tt:N',
    )

    # debug
    a_Pt = alt.Chart(d_Nodes).mark_point() \
    .encode(
        x = alt.X('x:Q', title=''),
        y = alt.Y('y:Q', title=''),
        size = alt.value(1),
    )

    # tri
    a_Tri = alt.Chart(d_Arcs).mark_point(
        filled  = True,
        size    = 300,
        shape   = 'triangle-up',
    ).encode(
        x       = alt.X('xe', type='quantitative', title=''),
        y       = alt.Y('ye', type='quantitative', title=''),
        angle   = alt.Angle('a:Q'),
        color   = alt.condition((alt.datum.sel) == 1, alt.value(color_arc_sel), alt.value(color_arc_des)),
        opacity = alt.condition((alt.datum.sel) == 1, alt.value(1.0 if p_arrows else 0.0), alt.value(0.0)),
        tooltip = 'tt:N',
    )


    with gr_cols[1]:
        if p_graph:
            st.altair_chart(
                (a_Arcs + \
                 a_Arrows + \
                 # a_Tri + \
                 a_Nodes + \
                 a_Labels + \
                 a_Descriptions + \
                 a_Text + \
                 a_Costs
                ).configure_axis(
                    grid    = False, 
                    domain  = False, 
                ).properties(
                    width  = s_graph[0],
                    height = s_graph[1],
                ), use_container_width=False)


# disable text area spell checks
html(js_text_areas, height=0)
# st.markdown(js_text_areas, unsafe_allow_html=True)

