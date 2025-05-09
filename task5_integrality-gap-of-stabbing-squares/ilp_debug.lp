\ Model lineStabILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x8b5feeb35618301d
Minimize
  h_1.5 + h_2.5 + v_1.5 + v_2.5
Subject To
 cover: h_1.5 + h_2.5 + v_1.5 + v_2.5 >= 1
 cover: v_1.5 + v_2.5 >= 1
 cover: h_1.5 + h_2.5 >= 1
 cover: >= 1
 cover: >= 1
 cover: >= 1
Bounds
Binaries
 h_1.5 h_2.5 v_1.5 v_2.5
End
