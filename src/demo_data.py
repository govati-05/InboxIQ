"""
InboxIQ - Built-in Demo Inbox
================================
A fixed, fully fictional sample inbox used by Demo Mode so the whole
application can be presented offline, with no Gmail account and no OAuth.

Every person, company, college, bank, order number and account number below
is invented. No real personal data is used anywhere in this file.

Each sample email carries a hand-assigned reference label
(`expected_category`, `expected_priority`, `expected_score`). These are NOT
what the dashboard shows when the ML models are available - in that case the
email text is run through the real prediction pipeline (src/predict.py) and
the reference labels are only used to measure how well the models agree with
a human labeller (see the "Demo Inbox Evaluation" section in ML Analytics).
The reference labels are only displayed directly as a fallback when the
models have not been trained yet.

Dates are stored as "hours before now", so the sample inbox always looks
fresh no matter when the demo is run.
"""

from datetime import datetime, timedelta

DEMO_ACCOUNT_NAME = "Aarav Kulkarni"
DEMO_ACCOUNT_EMAIL = "aarav.kulkarni@northbridge.edu"

# (sender_name, sender_email, subject, body, hours_ago,
#  expected_category, expected_priority, expected_score, is_read, is_starred)
_SAMPLE_EMAILS = [
    # ------------------------------------------------------------------ Education
    ("Prof. Kavitha Rao", "kavitha.rao@northbridge.edu",
     "Assignment deadline reminder for Machine Learning",
     "Dear students, this is a reminder that Assignment 3 (Decision Trees and Random Forests) "
     "for Machine Learning is due tomorrow by 11:59 PM. Submit your report and code on the "
     "course portal before the deadline. Late submissions will not be accepted.",
     3, "Education", "HIGH", 86, False, True),
    ("Examination Cell, Northbridge Institute", "examcell@northbridge.edu",
     "Your exam timetable has been released",
     "Dear student, the internal assessment II exam timetable for semester 6 has been released. "
     "Exams begin on Monday. Please check the timetable on the student portal and carry your "
     "college ID card to every exam.",
     9, "Education", "HIGH", 79, False, True),
    ("Scholarship Office", "scholarships@northbridge.edu",
     "Scholarship application open - Merit Scholarship 2026",
     "Dear student, applications for the Northbridge Merit Scholarship are now open for students "
     "with a CGPA above 8.5. Submit the application form with your mark sheets on the scholarship "
     "portal before the last date, 20th of this month.",
     26, "Education", "MEDIUM", 64, True, True),
    ("Department of CSE", "cse.office@northbridge.edu",
     "Class rescheduled: Compiler Design lab",
     "Dear students, the Compiler Design lab scheduled for Thursday has been rescheduled to Friday "
     "2:00 PM in Lab 4 due to a faculty meeting. Attendance will be taken as usual.",
     30, "Education", "MEDIUM", 52, True, False),
    ("Central Library", "library@northbridge.edu",
     "Library book return reminder",
     "Dear student, the book 'Pattern Recognition and Machine Learning' issued on your library card "
     "is due for return this week. Please return or renew it to avoid a late fine.",
     52, "Education", "MEDIUM", 48, True, False),
    ("Examination Cell, Northbridge Institute", "examcell@northbridge.edu",
     "Result declared for Database Management Systems",
     "Dear student, results for the Database Management Systems mid-semester exam have been "
     "declared. You can view your marks and grade on the student portal.",
     74, "Education", "MEDIUM", 55, True, False),
    ("Accounts Section", "accounts@northbridge.edu",
     "Semester fee payment due",
     "Dear student, this is a reminder that your semester fee payment is due by Friday. Please pay "
     "through the student portal before the deadline to avoid a late payment penalty.",
     100, "Education", "HIGH", 77, True, False),

    # ------------------------------------------------------------------ Work / internships
    ("Talent Team, Brightloop Technologies", "careers@brightloop-tech.com",
     "Your technical interview is scheduled",
     "Hi Aarav, thank you for applying for the Software Engineering Intern role at Brightloop "
     "Technologies. Your technical interview is scheduled for Wednesday at 11:00 AM over video call. "
     "Please confirm your availability by replying to this email.",
     5, "Work", "HIGH", 84, False, True),
    ("Neha Joshi", "neha.joshi@northbridge.edu",
     "Project status meeting tomorrow",
     "Hi team, our final year project status meeting is tomorrow at 4 PM in the seminar hall. "
     "Please bring your progress update on the model training and the dashboard module. "
     "Our guide will review the demo.",
     12, "Work", "MEDIUM", 66, False, False),
    ("Rohan Desai", "rohan.desai@brightloop-tech.com",
     "Offer letter attached - Summer Internship",
     "Dear Aarav, congratulations! Please find attached your offer letter for the Summer Internship "
     "at Brightloop Technologies. Kindly review, sign and return it within 3 days to confirm your joining.",
     58, "Work", "HIGH", 82, True, True),
    ("Campus Placement Cell", "placements@northbridge.edu",
     "Performance review scheduled for internship batch",
     "Hello, the mid-term performance review for the current internship batch is scheduled for next "
     "Tuesday. Please update your weekly report on the placement portal before the review.",
     80, "Work", "MEDIUM", 58, True, False),
    ("Ishaan Mehra", "ishaan.mehra@northbridge.edu",
     "Team lunch this Friday",
     "Hey everyone, we are planning a team lunch this Friday after the project review to celebrate "
     "finishing the prototype. Let me know if you can make it.",
     96, "Work", "MEDIUM", 44, True, False),
    ("Dr. Sameer Kulkarni", "sameer.k@northbridge.edu",
     "Action required: submit project synopsis by today",
     "Dear team, the final year project synopsis must be submitted on the portal by today 6 PM. "
     "This is important - groups that miss the deadline will have to present again next week.",
     2, "Work", "HIGH", 88, False, True),

    # ------------------------------------------------------------------ Finance
    ("Crestline Bank Alerts", "alerts@crestlinebank.example",
     "Debit alert: INR 2,450.00 spent from your account",
     "Dear customer, INR 2,450.00 has been debited from your Crestline Bank account XX4821 for a "
     "card transaction at Nimbus Mart. Available balance is INR 18,302.55. If you did not make this "
     "transaction, call our helpline immediately.",
     2, "Finance", "HIGH", 81, False, False),
    ("Crestline Bank", "statements@crestlinebank.example",
     "Your account statement is ready",
     "Dear customer, your monthly account statement for savings account XX4821 is now ready. "
     "Log in to net banking to view and download your statement.",
     40, "Finance", "MEDIUM", 50, True, False),
    ("Crestline Credit Cards", "cards@crestlinebank.example",
     "Your credit card payment is due",
     "Dear customer, the minimum amount due on your Crestline credit card ending 7730 is INR 1,200. "
     "Please pay before the due date on the 15th to avoid late fees and interest charges.",
     64, "Finance", "HIGH", 78, False, False),
    ("PayNest Wallet", "no-reply@paynest.example",
     "Money received in your wallet",
     "Hi Aarav, you have received INR 500.00 from Kabir Shah in your PayNest wallet. Your updated "
     "wallet balance is INR 1,140.00.",
     120, "Finance", "MEDIUM", 46, True, False),
    ("Crestline Bank Alerts", "alerts@crestlinebank.example",
     "Stipend credited to your account",
     "Dear customer, INR 15,000.00 has been credited to your Crestline Bank account XX4821 as "
     "internship stipend from Brightloop Technologies. Available balance is INR 33,302.55.",
     170, "Finance", "MEDIUM", 50, True, False),

    # ------------------------------------------------------------------ Shopping
    ("Nimbus Cart", "orders@nimbuscart.example",
     "Your order has been shipped",
     "Hi Aarav, your order #NC-584213 (Wireless Mouse and USB-C Hub) has been shipped and will be "
     "delivered in 2 days. Track your package from the Orders section of the app.",
     7, "Shopping", "LOW", 32, False, False),
    ("Nimbus Cart", "orders@nimbuscart.example",
     "Order confirmation #NC-584213",
     "Thank you for shopping with Nimbus Cart. Your order #NC-584213 has been placed successfully. "
     "Payment of INR 1,849 received. We will notify you when your order ships.",
     34, "Shopping", "LOW", 30, True, False),
    ("Urban Thread", "support@urbanthread.example",
     "Your order is out for delivery",
     "Good news! Your order #UT-30912 is out for delivery today. Please keep your phone available "
     "for the delivery partner.",
     15, "Shopping", "MEDIUM", 42, False, False),
    ("Bookhaven", "orders@bookhaven.example",
     "Invoice for your recent purchase",
     "Hi Aarav, thank you for your purchase of 'Hands-On Machine Learning'. Your invoice for order "
     "#BH-77120 is attached for your records.",
     140, "Shopping", "LOW", 22, True, False),
    ("Nimbus Cart", "alerts@nimbuscart.example",
     "Price drop alert on your wishlist item",
     "The price of 'Mechanical Keyboard K2' on your wishlist has dropped. Check it out before stock "
     "runs out.",
     160, "Shopping", "LOW", 24, True, False),

    # ------------------------------------------------------------------ Security
    ("Nimbus Account Security", "security@nimbuscart.example",
     "New login detected on your account",
     "We noticed a new login to your Nimbus Cart account from a Windows device in Pune. If this was "
     "you, no action is needed. If not, change your password immediately and review your security settings.",
     1, "Security", "HIGH", 93, False, False),
    ("Northbridge IT Services", "it-helpdesk@northbridge.edu",
     "Your password will expire in 3 days",
     "Dear user, your college portal password will expire in 3 days. Please change your password "
     "from the account security page to avoid being locked out of the student portal.",
     20, "Security", "HIGH", 88, False, False),
    ("Crestline Bank Security", "security@crestlinebank.example",
     "Two-factor authentication code",
     "Your one-time verification code for Crestline net banking login is 482913. This OTP is valid "
     "for 10 minutes. Never share this code with anyone.",
     44, "Security", "HIGH", 90, True, False),
    ("CloudBox Drive", "no-reply@cloudbox.example",
     "Unusual sign-in attempt blocked",
     "We blocked a suspicious sign-in attempt to your CloudBox account from an unrecognised device. "
     "We recommend you verify your recent activity and update your password.",
     88, "Security", "HIGH", 91, True, True),

    # ------------------------------------------------------------------ Personal
    ("Mom", "sunita.kulkarni.home@mailbox.example",
     "Can you call me when free?",
     "Hi beta, call me when you are free tonight. Need to discuss the plans for Diwali and your "
     "train tickets home.",
     4, "Personal", "MEDIUM", 60, False, True),
    ("Kabir Shah", "kabir.shah@mailbox.example",
     "Photos from last weekend",
     "Hey! Sharing the photos from our trek last weekend. The sunrise ones came out really well. "
     "Let me know which ones you want printed.",
     28, "Personal", "LOW", 34, True, False),
    ("Ananya Iyer", "ananya.iyer@mailbox.example",
     "Dinner plans for this week",
     "Hi Aarav, a few of us are planning dinner on Saturday to celebrate the end of exams. "
     "Are you in? Thinking of the new place near the campus gate.",
     48, "Personal", "MEDIUM", 45, True, False),
    ("Uncle Prakash", "prakash.k@mailbox.example",
     "Congratulations on your achievement",
     "Dear Aarav, heard from your father about the internship offer. Congratulations and best wishes! "
     "We are all very proud of you.",
     70, "Personal", "LOW", 36, True, True),

    # ------------------------------------------------------------------ Promotions
    ("StyleKart", "offers@stylekart.example",
     "Flash sale: Up to 70% off",
     "Mega flash sale is live now! Get up to 70% off on top fashion brands. Limited stock. "
     "Shop now and use code STYLE70 at checkout.",
     6, "Promotions", "LOW", 12, False, False),
    ("Nimbus Cart Offers", "offers@nimbuscart.example",
     "Last chance to save 40%",
     "Hurry! Last chance to save 40% on laptops, headphones and accessories. Offer ends tonight. "
     "Exclusive deal for our members.",
     22, "Promotions", "LOW", 15, True, False),
    ("FoodieGo", "deals@foodiego.example",
     "Buy one get one free offer",
     "Craving something tasty? Enjoy a buy one get one free offer on all pizzas this weekend. "
     "Order on the FoodieGo app to grab the deal.",
     54, "Promotions", "LOW", 10, True, False),
    ("CodeCamp Academy", "hello@codecamp.example",
     "Exclusive deal just for you",
     "Upgrade your skills with our Data Science bootcamp. Enroll this week and get an exclusive "
     "50% discount on the full course.",
     110, "Promotions", "LOW", 9, True, False),

    # ------------------------------------------------------------------ Social
    ("LinkSphere", "notifications@linksphere.example",
     "Rohan Desai sent you a connection request",
     "Rohan Desai, Engineering Manager at Brightloop Technologies, sent you a connection request. "
     "Accept the request to grow your network.",
     11, "Social", "LOW", 18, False, False),
    ("Snapgram", "no-reply@snapgram.example",
     "Kabir tagged you in a post",
     "Kabir Shah tagged you in a post: 'Trek squad on top of the hill'. See the post and leave a comment.",
     29, "Social", "LOW", 14, True, False),
    ("ChirpNet", "notify@chirpnet.example",
     "You have a new follower",
     "Ananya Iyer started following you. Follow back to stay connected and see their updates.",
     76, "Social", "LOW", 10, True, False),
    ("Snapgram", "no-reply@snapgram.example",
     "Weekly activity summary",
     "Here is your weekly activity summary: 12 new likes, 3 new comments and 2 new followers on your "
     "posts this week.",
     150, "Social", "LOW", 8, True, False),

    # ------------------------------------------------------------------ Travel / Events / Newsletter
    ("RailYatra", "tickets@railyatra.example",
     "Your train ticket is confirmed",
     "Your train ticket from Pune to Nagpur on the 28th is confirmed. PNR 4821937710, coach B2, "
     "berth 34. Please carry a valid ID proof during the journey.",
     18, "Travel", "MEDIUM", 62, False, False),
    ("SkyWays Airlines", "checkin@skyways.example",
     "Web check-in is now open for your flight",
     "Dear passenger, web check-in is now open for flight SW 612 from Pune to Bengaluru departing "
     "tomorrow at 7:10 AM. Check in online to select your seat.",
     36, "Travel", "HIGH", 75, True, False),
    ("Northbridge Coding Club", "codingclub@northbridge.edu",
     "Hackathon registration is open",
     "Registrations for HackNorth 2026, our 24-hour campus hackathon, are now open. Form teams of up "
     "to 4 and register before Sunday to secure your spot.",
     24, "Events", "MEDIUM", 50, False, True),
    ("Developer Circle Pune", "events@devcircle.example",
     "Invitation: AI and ML meetup this Saturday",
     "You are invited to our monthly AI and ML community meetup this Saturday at 5 PM. Talks on "
     "transformers and MLOps, followed by networking. RSVP to confirm your seat.",
     66, "Events", "MEDIUM", 40, True, False),
    ("The Weekly Byte", "newsletter@weeklybyte.example",
     "This week in tech: AI, cloud and open source",
     "Welcome to this week's edition of The Weekly Byte newsletter. Top stories: new open source "
     "language models, cloud cost tips, and a beginner's guide to vector databases. Read more inside.",
     46, "Newsletter", "LOW", 8, True, False),
    ("Campus Chronicle", "chronicle@northbridge.edu",
     "Campus Chronicle - monthly newsletter",
     "In this month's newsletter: highlights from the annual tech fest, new research lab inaugurated, "
     "and alumni success stories. Subscribe to get every issue.",
     130, "Newsletter", "LOW", 6, True, False),
]


def get_demo_emails(now=None):
    """
    Return the demo inbox as a list of dicts using the same field names as
    the SQLite `emails` table (so it plugs into the existing dashboard code),
    plus a few demo-only fields: sender_name, sender_email, is_read,
    is_starred and the reference labels expected_*.

    `category`, `priority`, `priority_score` and `confidence` start as None;
    app.py fills them in by running the real ML models.
    """
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    emails = []
    for i, (name, addr, subject, body, hours_ago, cat, pri, score, is_read, is_starred) in enumerate(_SAMPLE_EMAILS):
        labels = ["INBOX"]
        if not is_read:
            labels.append("UNREAD")
        if is_starred:
            labels.append("STARRED")
        emails.append({
            "message_id": f"demo-{i + 1:03d}",
            "thread_id": f"demo-thread-{i + 1:03d}",
            "sender": f"{name} <{addr}>",
            "sender_name": name,
            "sender_email": addr,
            "recipient": f"{DEMO_ACCOUNT_NAME} <{DEMO_ACCOUNT_EMAIL}>",
            "subject": subject,
            "body": body,
            "snippet": body[:140],
            "date": (now - timedelta(hours=hours_ago)).isoformat(),
            "labels": ",".join(labels),
            "is_read": is_read,
            "is_starred": is_starred,
            "expected_category": cat,
            "expected_priority": pri,
            "expected_score": score,
            "category": None,
            "priority": None,
            "priority_score": None,
            "confidence": None,
        })
    return emails
