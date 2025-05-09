\ Model lineStabLP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x8b5feeb35618301d
Minimize
  h_1.5 + h_2.5 + v_1.5 + v_2.5
Subject To
 R0: h_1.5 + h_2.5 + v_1.5 + v_2.5 >= 1
 R1: v_1.5 + v_2.5 >= 1
 R2: h_1.5 + h_2.5 >= 1
 R3: >= 1
 R4: >= 1
 R5: >= 1
Bounds
 h_1.5 <= 1
 h_2.5 <= 1
 v_1.5 <= 1
 v_2.5 <= 1
End
