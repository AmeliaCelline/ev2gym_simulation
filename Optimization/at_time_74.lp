\ Model optimization_model
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xd747d43e4b7ee448
Minimize
  48.87815361255879 evs_power[0,0] + 52.28164505453232 evs_power[1,0]
   + 56.88140957427678 evs_power[2,0] + 58.24278097850058 evs_power[3,0]
   + 59.31767413026468 evs_power[4,0] + 62.6049677290689 evs_power[5,0]
   + 62.33609689761114 evs_power[6,0] + 63.58907668449605 evs_power[7,0]
   + 64.66027968252763 evs_power[8,0] + 63.84959653527297 evs_power[9,0]
   + 63.52282298845545 evs_power[10,0] + 62.9097896318073 evs_power[11,0]
   + 63.43463185167988 evs_power[12,0] + 62.84795635289653 evs_power[13,0]
   + 59.92661466833688 evs_power[14,0] + 59.53333429652307 evs_power[15,0]
   + 58.36327969914191 evs_power[16,0] + 56.64766134876361 evs_power[17,0]
   + 54.49632506011841 evs_power[18,0] + 52.47471512994433 evs_power[19,0]
   + 50.25797040370812 evs_power[20,0] + 46.90797266083499 evs_power[21,0]
   + 47.65953753020082 evs_power[22,0] + 44.95029664538522 evs_power[23,0]
   + 41.96723912314291 evs_power[24,0] + 39.96352298668069 evs_power[25,0]
   + 35.22972396920257 evs_power[26,0] + 32.87819672494571 evs_power[27,0]
   + 29.93959603283348 evs_power[28,0] + 25.12067218346802 evs_power[29,0]
   + 21.41918322298142 evs_power[30,0] + 21.70043949839084 evs_power[31,0]
   + 16.54348817015985 evs_power[32,0] + 13.57335279694819 evs_power[33,0]
   + 9.140485855949606 evs_power[34,0] + 4.618406022532819 evs_power[35,0]
   + 5.325137697390661 evs_power[36,0] - 0.5707982896096311 evs_power[37,0]
   - 2.010450811748214 evs_power[38,0] - 1.880059203275877 evs_power[39,0]
   - 6.619523080100677 evs_power[40,0] - 6.938725328824475 evs_power[41,0]
   - 9.979554230850255 evs_power[42,0] - 12.09212595318699 evs_power[43,0]
   - 13.36128285757887 evs_power[44,0] - 15.13126759856531 evs_power[45,0]
   - 17.33543724648557 evs_power[46,0] - 15.21877657648257 evs_power[47,0]
   - 19.78640649966115 evs_power[48,0] - 20.81658318606548 evs_power[49,0]
   - 20.43995904482627 evs_power[50,0] - 21.5540340344122 evs_power[51,0]
   - 21.5717499155293 evs_power[52,0] - 20.90816755438364 evs_power[53,0]
   - 21.31480613671705 evs_power[54,0] - 19.51238649870451 evs_power[55,0]
   - 19.23094397316774 evs_power[56,0] - 19.59169632950546 evs_power[57,0]
   - 18.5567136561437 evs_power[58,0] - 17.95865308065925 evs_power[59,0]
   - 17.03316946170511 evs_power[60,0] - 16.04195349135312 evs_power[61,0]
   - 16.7390393878891 evs_power[62,0] - 10.94193374763005 evs_power[63,0]
   - 9.394472889922753 evs_power[64,0] - 9.369307199248373 evs_power[65,0]
   - 7.725158625442085 evs_power[66,0] - 4.385627167756169 evs_power[67,0]
   - 4.174712428121737 evs_power[68,0] - 1.370013288453402 evs_power[69,0]
   + 2.522141889443051 evs_power[70,0] + 5.152925396638128 evs_power[71,0]
   + 6.815133394608896 evs_power[72,0] + 8.467002998747773 evs_power[73,0]
   + 6.513815636422851 evs_power[74,0] + 10.30845783468618 evs_power[75,0]
   + 13.26039229334411 evs_power[76,0] + 12.06639584507161 evs_power[77,0]
   + 16.22763001705344 evs_power[78,0] + 17.59755890925254 evs_power[79,0]
   + 17.54412614322094 evs_power[80,0] + 18.87121923235571 evs_power[81,0]
   + 18.42159007461719 evs_power[82,0] + 20.73001291419425 evs_power[83,0]
   + 20.3291168005983 evs_power[84,0] + 18.30301340584383 evs_power[85,0]
   + 22.53904217667802 evs_power[86,0] + 21.31217598582759 evs_power[87,0]
   + 21.72109930811311 evs_power[88,0] + 20.42060977107123 evs_power[89,0]
   + 20.60639388238127 evs_power[90,0] + 21.282365868097 evs_power[91,0]
   + 21.26419835327735 evs_power[92,0] + 22.25010227716552 evs_power[93,0]
   + 21.66658867975 evs_power[94,0] + 23.28866540749398 evs_power[95,0]
   + 25999.52961951741 Constant + [ 2 evs_power[0,0] ^2
   + 2 evs_power[1,0] ^2 + 2 evs_power[2,0] ^2 + 2 evs_power[3,0] ^2
   + 2 evs_power[4,0] ^2 + 2 evs_power[5,0] ^2 + 2 evs_power[6,0] ^2
   + 2 evs_power[7,0] ^2 + 2 evs_power[8,0] ^2 + 2 evs_power[9,0] ^2
   + 2 evs_power[10,0] ^2 + 2 evs_power[11,0] ^2 + 2 evs_power[12,0] ^2
   + 2 evs_power[13,0] ^2 + 2 evs_power[14,0] ^2 + 2 evs_power[15,0] ^2
   + 2 evs_power[16,0] ^2 + 2 evs_power[17,0] ^2 + 2 evs_power[18,0] ^2
   + 2 evs_power[19,0] ^2 + 2 evs_power[20,0] ^2 + 2 evs_power[21,0] ^2
   + 2 evs_power[22,0] ^2 + 2 evs_power[23,0] ^2 + 2 evs_power[24,0] ^2
   + 2 evs_power[25,0] ^2 + 2 evs_power[26,0] ^2 + 2 evs_power[27,0] ^2
   + 2 evs_power[28,0] ^2 + 2 evs_power[29,0] ^2 + 2 evs_power[30,0] ^2
   + 2 evs_power[31,0] ^2 + 2 evs_power[32,0] ^2 + 2 evs_power[33,0] ^2
   + 2 evs_power[34,0] ^2 + 2 evs_power[35,0] ^2 + 2 evs_power[36,0] ^2
   + 2 evs_power[37,0] ^2 + 2 evs_power[38,0] ^2 + 2 evs_power[39,0] ^2
   + 2 evs_power[40,0] ^2 + 2 evs_power[41,0] ^2 + 2 evs_power[42,0] ^2
   + 2 evs_power[43,0] ^2 + 2 evs_power[44,0] ^2 + 2 evs_power[45,0] ^2
   + 2 evs_power[46,0] ^2 + 2 evs_power[47,0] ^2 + 2 evs_power[48,0] ^2
   + 2 evs_power[49,0] ^2 + 2 evs_power[50,0] ^2 + 2 evs_power[51,0] ^2
   + 2 evs_power[52,0] ^2 + 2 evs_power[53,0] ^2 + 2 evs_power[54,0] ^2
   + 2 evs_power[55,0] ^2 + 2 evs_power[56,0] ^2 + 2 evs_power[57,0] ^2
   + 2 evs_power[58,0] ^2 + 2 evs_power[59,0] ^2 + 2 evs_power[60,0] ^2
   + 2 evs_power[61,0] ^2 + 2 evs_power[62,0] ^2 + 2 evs_power[63,0] ^2
   + 2 evs_power[64,0] ^2 + 2 evs_power[65,0] ^2 + 2 evs_power[66,0] ^2
   + 2 evs_power[67,0] ^2 + 2 evs_power[68,0] ^2 + 2 evs_power[69,0] ^2
   + 2 evs_power[70,0] ^2 + 2 evs_power[71,0] ^2 + 2 evs_power[72,0] ^2
   + 2 evs_power[73,0] ^2 + 2 evs_power[74,0] ^2 + 2 evs_power[75,0] ^2
   + 2 evs_power[76,0] ^2 + 2 evs_power[77,0] ^2 + 2 evs_power[78,0] ^2
   + 2 evs_power[79,0] ^2 + 2 evs_power[80,0] ^2 + 2 evs_power[81,0] ^2
   + 2 evs_power[82,0] ^2 + 2 evs_power[83,0] ^2 + 2 evs_power[84,0] ^2
   + 2 evs_power[85,0] ^2 + 2 evs_power[86,0] ^2 + 2 evs_power[87,0] ^2
   + 2 evs_power[88,0] ^2 + 2 evs_power[89,0] ^2 + 2 evs_power[90,0] ^2
   + 2 evs_power[91,0] ^2 + 2 evs_power[92,0] ^2 + 2 evs_power[93,0] ^2
   + 2 evs_power[94,0] ^2 + 2 evs_power[95,0] ^2 ] / 2 
Subject To
 power==0_EV_Departed_time0_counter0: evs_power[0,0] = 0
 power==0_EV_Departed_time1_counter0: evs_power[1,0] = 0
 power==0_EV_Departed_time2_counter0: evs_power[2,0] = 0
 power==0_EV_Departed_time3_counter0: evs_power[3,0] = 0
 power==0_EV_Departed_time4_counter0: evs_power[4,0] = 0
 power==0_EV_Departed_time5_counter0: evs_power[5,0] = 0
 power==0_EV_Departed_time6_counter0: evs_power[6,0] = 0
 power==0_EV_Departed_time7_counter0: evs_power[7,0] = 0
 power==0_EV_Departed_time8_counter0: evs_power[8,0] = 0
 power==0_EV_Departed_time9_counter0: evs_power[9,0] = 0
 power==0_EV_Departed_time10_counter0: evs_power[10,0] = 0
 power==0_EV_Departed_time11_counter0: evs_power[11,0] = 0
 power==0_EV_Departed_time12_counter0: evs_power[12,0] = 0
 power==0_EV_Departed_time13_counter0: evs_power[13,0] = 0
 power==0_EV_Departed_time14_counter0: evs_power[14,0] = 0
 power==0_EV_Departed_time15_counter0: evs_power[15,0] = 0
 power==0_EV_Departed_time16_counter0: evs_power[16,0] = 0
 power==0_EV_Departed_time17_counter0: evs_power[17,0] = 0
 power==0_EV_Departed_time18_counter0: evs_power[18,0] = 0
 power==0_EV_Departed_time19_counter0: evs_power[19,0] = 0
 power==0_EV_Departed_time20_counter0: evs_power[20,0] = 0
 power==0_EV_Departed_time21_counter0: evs_power[21,0] = 0
 power==0_EV_Departed_time22_counter0: evs_power[22,0] = 0
 power==0_EV_Departed_time23_counter0: evs_power[23,0] = 0
 power==0_EV_Departed_time24_counter0: evs_power[24,0] = 0
 power==0_EV_Departed_time25_counter0: evs_power[25,0] = 0
 power==0_EV_Departed_time26_counter0: evs_power[26,0] = 0
 power==0_EV_Departed_time27_counter0: evs_power[27,0] = 0
 power==0_EV_Departed_time28_counter0: evs_power[28,0] = 0
 power==0_EV_Departed_time29_counter0: evs_power[29,0] = 0
 power==0_EV_Departed_time30_counter0: evs_power[30,0] = 0
 power==0_EV_Departed_time31_counter0: evs_power[31,0] = 0
 power==0_EV_Departed_time32_counter0: evs_power[32,0] = 0
 power==0_EV_Departed_time33_counter0: evs_power[33,0] = 0
 power==0_EV_Departed_time34_counter0: evs_power[34,0] = 0
 power==0_EV_Departed_time35_counter0: evs_power[35,0] = 0
 power==0_EV_Departed_time36_counter0: evs_power[36,0] = 0
 power==0_EV_Departed_time37_counter0: evs_power[37,0] = 0
 power==0_EV_Departed_time38_counter0: evs_power[38,0] = 0
 power==0_EV_Departed_time39_counter0: evs_power[39,0] = 0
 power==0_EV_Departed_time40_counter0: evs_power[40,0] = 0
 power==0_EV_Departed_time41_counter0: evs_power[41,0] = 0
 power==0_EV_Departed_time42_counter0: evs_power[42,0] = 0
 power==0_EV_Departed_time43_counter0: evs_power[43,0] = 0
 power==0_EV_Departed_time44_counter0: evs_power[44,0] = 0
 power==0_EV_Departed_time45_counter0: evs_power[45,0] = 0
 power==0_EV_Departed_time46_counter0: evs_power[46,0] = 0
 power==0_EV_Departed_time47_counter0: evs_power[47,0] = 0
 power==0_EV_Departed_time48_counter0: evs_power[48,0] = 0
 power==0_EV_Departed_time49_counter0: evs_power[49,0] = 0
 power==0_EV_Departed_time50_counter0: evs_power[50,0] = 0
 power==0_EV_Departed_time51_counter0: evs_power[51,0] = 0
 power==0_EV_Departed_time52_counter0: evs_power[52,0] = 0
 power==0_EV_Departed_time53_counter0: evs_power[53,0] = 0
 power==0_EV_Departed_time54_counter0: evs_power[54,0] = 0
 power==0_EV_Departed_time55_counter0: evs_power[55,0] = 0
 power==0_EV_Departed_time56_counter0: evs_power[56,0] = 0
 power==0_EV_Departed_time57_counter0: evs_power[57,0] = 0
 power==0_EV_Departed_time58_counter0: evs_power[58,0] = 0
 power==0_EV_Departed_time59_counter0: evs_power[59,0] = 0
 power==0_EV_Departed_time60_counter0: evs_power[60,0] = 0
 power==0_EV_Departed_time61_counter0: evs_power[61,0] = 0
 power==0_EV_Departed_time62_counter0: evs_power[62,0] = 0
 power==0_EV_Departed_time63_counter0: evs_power[63,0] = 0
 power==0_EV_Departed_time64_counter0: evs_power[64,0] = 0
 power==0_EV_Departed_time65_counter0: evs_power[65,0] = 0
 power==0_EV_Departed_time66_counter0: evs_power[66,0] = 0
 power==0_EV_Departed_time67_counter0: evs_power[67,0] = 0
 power==0_EV_Departed_time68_counter0: evs_power[68,0] = 0
 power==0_EV_Departed_time69_counter0: evs_power[69,0] = 0
 power==0_EV_Departed_time70_counter0: evs_power[70,0] = 0
 power==0_EV_Departed_time71_counter0: evs_power[71,0] = 0
 power==0_EV_Departed_time72_counter0: evs_power[72,0] = 0
 power==0_EV_Departed_time73_counter0: evs_power[73,0] = 0
 power==0_EV_Departed_time74_counter0: evs_power[74,0] = 0
 power==0_EV_Departed_time75_counter0: evs_power[75,0] = 0
 power==0_EV_Departed_time76_counter0: evs_power[76,0] = 0
 power==0_EV_Departed_time77_counter0: evs_power[77,0] = 0
 power==0_EV_Departed_time78_counter0: evs_power[78,0] = 0
 power==0_EV_Departed_time79_counter0: evs_power[79,0] = 0
 power==0_EV_Departed_time80_counter0: evs_power[80,0] = 0
 power==0_EV_Departed_time81_counter0: evs_power[81,0] = 0
 power==0_EV_Departed_time82_counter0: evs_power[82,0] = 0
 power==0_EV_Departed_time83_counter0: evs_power[83,0] = 0
 power==0_EV_Departed_time84_counter0: evs_power[84,0] = 0
 power==0_EV_Departed_time85_counter0: evs_power[85,0] = 0
 power==0_EV_Departed_time86_counter0: evs_power[86,0] = 0
 power==0_EV_Departed_time87_counter0: evs_power[87,0] = 0
 power==0_EV_Departed_time88_counter0: evs_power[88,0] = 0
 power==0_EV_Departed_time89_counter0: evs_power[89,0] = 0
 power==0_EV_Departed_time90_counter0: evs_power[90,0] = 0
 power==0_EV_Departed_time91_counter0: evs_power[91,0] = 0
 power==0_EV_Departed_time92_counter0: evs_power[92,0] = 0
 power==0_EV_Departed_time93_counter0: evs_power[93,0] = 0
 power==0_EV_Departed_time94_counter0: evs_power[94,0] = 0
 power==0_EV_Departed_time95_counter0: evs_power[95,0] = 0
Bounds
 -22.30881440148714 <= evs_power[0,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[1,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[2,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[3,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[4,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[5,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[6,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[7,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[8,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[9,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[10,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[11,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[12,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[13,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[14,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[15,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[16,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[17,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[18,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[19,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[20,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[21,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[22,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[23,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[24,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[25,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[26,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[27,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[28,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[29,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[30,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[31,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[32,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[33,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[34,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[35,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[36,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[37,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[38,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[39,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[40,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[41,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[42,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[43,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[44,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[45,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[46,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[47,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[48,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[49,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[50,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[51,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[52,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[53,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[54,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[55,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[56,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[57,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[58,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[59,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[60,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[61,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[62,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[63,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[64,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[65,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[66,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[67,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[68,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[69,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[70,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[71,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[72,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[73,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[74,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[75,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[76,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[77,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[78,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[79,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[80,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[81,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[82,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[83,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[84,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[85,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[86,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[87,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[88,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[89,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[90,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[91,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[92,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[93,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[94,0] <= 22.30881440148714
 -22.30881440148714 <= evs_power[95,0] <= 22.30881440148714
 Constant = 1
End
